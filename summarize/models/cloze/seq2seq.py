import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, MatrixAttention, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Any, Dict, List, Optional

from summarize.models.sds import Seq2SeqModel
from summarize.modules.bridge import Bridge
from summarize.modules.rnns import RNN
from summarize.nn.beam_search import BeamSearch


@Model.register('cloze-seq2seq')
class ClozeSeq2SeqModel(Seq2SeqModel):
    """
    An implementation of a standard encoder-decoder network with attention
    built on RNNs. For most parameter documentation, see ``Seq2SeqModel``.

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
                 attention: MatrixAttention,
                 attention_layer: FeedForward,
                 decoder: RNN,
                 bridge: Bridge,
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
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab=vocab,
                         document_token_embedder=document_token_embedder,
                         encoder=encoder,
                         attention=attention,
                         attention_layer=attention_layer,
                         decoder=decoder,
                         bridge=bridge,
                         beam_search=beam_search,
                         run_beam_search=run_beam_search,
                         summary_token_embedder=cloze_token_embedder,
                         summary_namespace=cloze_namespace,
                         use_input_feeding=use_input_feeding,
                         input_feeding_projection_layer=input_feeding_projection_layer,
                         instance_loss_normalization=instance_loss_normalization,
                         batch_loss_normalization=batch_loss_normalization,
                         metrics=metrics,
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
                                       state: Dict[str, torch.Tensor]):
        """
        Forces the context tokens through the decoder to prime the decoder
        for generating the cloze.
        """
        # shape: (batch_size, num_context_tokens)
        mask = get_text_field_mask(context)
        # shape: (batch_size, num_context_tokens)
        context_tokens = context['tokens']
        _, next_state = self._decoder_step(context_tokens, state, token_mask=mask)
        return next_state

    @overrides
    def forward(self,
                document: Dict[str, torch.Tensor],
                topics: Dict[str, torch.Tensor],
                context: Dict[str, torch.Tensor],
                metadata: List[Dict[str, Any]],
                cloze: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, num_document_tokens, encoder_hidden_size)
        # shape: (batch_size, num_document_tokens)
        # shape: (batch_size, decoder_hidden_size)
        # shape: (batch_size, decoder_hidden_size)
        encoder_outputs, document_mask, hidden, memory = \
            self._run_encoder(document)

        # The ``input_feed`` vector will not be used unless input feeding is enabled.
        # Initially, it will be all 0s.
        # shape: (batch_size, decoder_hidden_size)
        input_feed = encoder_outputs.new_zeros(encoder_outputs.size(0), self.attention_layer.get_output_dim())

        # Setup the state which will be used to initialize decoding
        initial_decoding_state = {
            'document_mask': document_mask,
            'encoder_outputs': encoder_outputs,
            'hidden': hidden,
            'memory': memory,
            'input_feed': input_feed
        }

        # If we use the context, then we need to update the decoder's hidden
        # state based on the context tokens
        if self.use_context:
            initial_decoding_state = self._force_context_through_decoder(context, initial_decoding_state)

        output_dict = {}

        # Compute the loss if we have the ground-truth summaries
        if cloze is not None:
            # shape: (batch_size, num_summary_tokens - 1, summary_vocab_size)
            # shape: (batch_size, num_summary_tokens - 1, summary_vocab_size)
            logits, targets = self._run_teacher_forcing(initial_decoding_state, cloze)
            output_dict['loss'] = self._compute_loss(logits, targets)

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
