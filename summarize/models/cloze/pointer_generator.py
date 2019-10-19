import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Any, Dict, List, Optional

from summarize.models.sds import PointerGeneratorModel
from summarize.modules.bridge import Bridge
from summarize.modules.coverage_matrix_attention import CoverageMatrixAttention
from summarize.modules.generate_probability_functions import GenerateProbabilityFunction
from summarize.modules.rnns import RNN
from summarize.nn.beam_search import BeamSearch


@Model.register('cloze-pointer-generator')
class ClozePointerGeneratorModel(PointerGeneratorModel):
    """
    An implementation of a the Pointer-Generator model for a cloze model. For
    most parameter documentation, see ``PointerGeneratorModel``.

    Parameters
    ----------
    use_context: bool
        Indicates whether or not the context should be used to initialize
        the decoder's state.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 document_token_embedder: TextFieldEmbedder,
                 encoder: RNN,
                 attention: CoverageMatrixAttention,
                 attention_layer: FeedForward,
                 decoder: RNN,
                 bridge: Bridge,
                 generate_probability_function: GenerateProbabilityFunction,
                 beam_search: BeamSearch,
                 use_context: bool,
                 run_beam_search: bool = True,
                 cloze_token_embedder: Optional[TokenEmbedder] = None,
                 cloze_namespace: str = 'tokens',
                 use_input_feeding: bool = False,
                 input_feeding_projection_layer: Optional[FeedForward] = None,
                 instance_loss_normalization: str = 'sum',
                 batch_loss_normalization: str = 'average',
                 metrics: Optional[List[Metric]] = None,
                 coverage_loss_weight: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab=vocab,
                         document_token_embedder=document_token_embedder,
                         encoder=encoder,
                         attention=attention,
                         attention_layer=attention_layer,
                         decoder=decoder,
                         bridge=bridge,
                         generate_probability_function=generate_probability_function,
                         beam_search=beam_search,
                         run_beam_search=run_beam_search,
                         summary_token_embedder=cloze_token_embedder,
                         summary_namespace=cloze_namespace,
                         use_input_feeding=use_input_feeding,
                         input_feeding_projection_layer=input_feeding_projection_layer,
                         instance_loss_normalization=instance_loss_normalization,
                         batch_loss_normalization=batch_loss_normalization,
                         metrics=metrics,
                         coverage_loss_weight=coverage_loss_weight,
                         initializer=initializer,
                         regularizer=regularizer)
        self.use_context = use_context

        # Forcing the context through the decoder and using input feeding will require
        # special processing. Forcing the context through the decoder requires passing
        # a token mask through the RNN, and input feeding requires writing a manual
        # for loop. If the contexts within one batch are of different lengths, the
        # for loop will eventually try to pass a sequence of length 0 through the RNN,
        # which fails. To enable using both of these options, we need to address this problem.
        if self.use_context and self.use_input_feeding:
            raise Exception('Using both ``use_context`` and ``use_input_feeding`` is not supported')

    def _force_context_through_decoder(self,
                                       context: Dict[str, torch.Tensor],
                                       context_token_document_indices_mask: torch.Tensor,
                                       state: Dict[str, torch.Tensor]):
        # Unpack everything from the input state
        # shape: (group_size, num_document_tokens)
        document_mask = state['document_mask']
        # shape: (group_size, num_document_tokens, encoder_hidden_size)
        encoder_outputs = state['encoder_outputs']
        # shape: (group_size, decoder_hidden_size)
        hidden = state['hidden']
        # shape: (group_size, decoder_hidden_size)
        memory = state['memory']
        # shape: (group_size, num_document_tokens)
        coverage = state['coverage']

        # shape: (batch_size, num_context_tokens)
        token_mask = get_text_field_mask(context)

        # shape: (batch_size, num_context_tokens)
        cloze_input_tokens = context['tokens']
        # shape: (group_size, num_summary_tokens, embedding_size)
        cloze_token_embeddings = self.summary_token_embedder(cloze_input_tokens)

        # Prepare the decoder hidden state. The hidden state expects the first
        # dimension to be the number of layers.
        # shape: (1, group_size, decoder_hidden_size)
        hidden = hidden.unsqueeze(0)
        if self.decoder.has_memory():
            # shape: (1, group_size, decoder_hidden_size)
            memory = memory.unsqueeze(0)
            decoder_state = (hidden, memory)
        else:
            decoder_state = hidden

        assert not self.use_input_feeding
        # shape: (1, group_size, decoder_hidden_size)
        # shape: (group-size, num_summary_tokens, num_document_tokens)
        # shape: (group-size, num_document_tokens)
        (_, _, decoder_state, _, _, coverage_vectors, coverage) = \
            self._decoder_forward(cloze_token_embeddings, decoder_state, encoder_outputs, document_mask, coverage, token_mask)

        # We need to pick the latest coverage vector, which will vary based
        # on the length of the input context. We don't have to do this in the
        # normal decoding step because we don't care about the values after
        # the end of the decoder inputs.
        # shape: (batch_size, 1)
        context_lengths = token_mask.sum(dim=1).long().unsqueeze(1)
        # shape: (batch_size, num_context_tokens + 1, num_document_tokens)
        coverage_vectors = torch.cat([coverage_vectors, coverage.unsqueeze(1)], dim=1)
        # shape: (batch_size, 1, num_document_tokens)
        context_lengths = context_lengths.unsqueeze(2).expand(-1, -1, coverage_vectors.size(2))
        # shape: (batch_size, num_document_tokens)
        coverage = coverage_vectors.gather(1, context_lengths).squeeze(1)

        # Remove the first dimension which is unnecessary as this is only
        # implemented for 1 layer.
        if self.decoder.has_memory():
            hidden, memory = decoder_state
            # shape: (group_size, decoder_hidden_size)
            hidden = hidden.squeeze(0)
            # shape: (group_size, decoder_hidden_size)
            memory = memory.squeeze(0)
        else:
            # shape: (group_size, decoder_hidden_size)
            hidden = hidden.squeeze(0)

        return hidden, memory, coverage

    @overrides
    def forward(self,
                document: Dict[str, torch.Tensor],
                document_token_first_indices: torch.Tensor,
                document_in_cloze_namespace: torch.Tensor,
                topics: Dict[str, torch.Tensor],
                context: Dict[str, torch.Tensor],
                context_token_document_indices: torch.Tensor,
                context_token_document_indices_mask: torch.Tensor,
                metadata: List[Dict[str, Any]],
                cloze: Optional[Dict[str, torch.Tensor]] = None,
                cloze_token_document_indices: Optional[torch.Tensor] = None,
                cloze_token_document_indices_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, num_document_tokens, encoder_hidden_size)
        # shape: (batch_size, num_document_tokens)
        # shape: (batch_size, decoder_hidden_size)
        # shape: (batch_size, decoder_hidden_size)
        encoder_outputs, document_mask, hidden, memory = \
            self._run_encoder(document)

        batch_size, num_document_tokens, _ = encoder_outputs.size()

        # The ``input_feed`` vector will not be used unless input feeding is enabled.
        # Initially, it will be all 0s.
        # shape: (batch_size, decoder_hidden_size)
        input_feed = encoder_outputs.new_zeros(batch_size, self.attention_layer.get_output_dim())

        # This will keep track of the accumulated coverage vector, which keeps track
        # of how much attention probability has been assigned to each document token
        # across decoding steps.
        # shape: (batch_size, num_document_tokens)
        coverage = encoder_outputs.new_zeros(batch_size, num_document_tokens)

        # Some of the tensors need to be converted to longs because they
        # represent indices
        document_token_first_indices = document_token_first_indices.long()
        document_in_cloze_namespace = document_in_cloze_namespace.long()

        # Setup the state which will be used to initialize decoding
        initial_decoding_state = {
            'document_mask': document_mask,
            'encoder_outputs': encoder_outputs,
            'hidden': hidden,
            'memory': memory,
            'input_feed': input_feed,
            'coverage': coverage,
            'document_token_first_indices': document_token_first_indices,
            'document_in_summary_namespace': document_in_cloze_namespace
        }

        # If we use the context, then we need to update the decoder's hidden
        # state based on the context tokens
        if self.use_context:
            hidden, memory, coverage = \
                self._force_context_through_decoder(context, context_token_document_indices_mask, initial_decoding_state)
            initial_decoding_state['hidden'] = hidden
            initial_decoding_state['memory'] = memory
            initial_decoding_state['coverage'] = coverage

        output_dict = {}

        # Compute the loss if we have the ground-truth summaries
        if cloze is not None:
            cloze_token_document_indices = cloze_token_document_indices.long()

            output_dict['loss'] = self._compute_loss(initial_decoding_state, cloze,
                                                     cloze_token_document_indices,
                                                     cloze_token_document_indices_mask)

        # If we aren't training, then we need to do inference
        if not self.training and self.run_beam_search:
            # shape: (batch_size, beam_size, max_output_length)
            # shape: (batch_size, beam_size)
            predictions, log_probabilities = self._run_inference(initial_decoding_state)
            output_dict['predictions'] = predictions
            output_dict['log_probabilities'] = log_probabilities
            self._update_metrics(predictions, metadata, summary_field_name='cloze')

        # Always include the metadata in the output dictionary
        output_dict['metadata'] = metadata
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return super().decode(output_dict, summary_field_name='cloze')
