import torch
import torch.nn.functional as F
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Any, Dict, List, Optional, Tuple, Union

from summarize.common.util import SENT_START_SYMBOL, SENT_END_SYMBOL
from summarize.modules.bridge import Bridge
from summarize.modules.coverage_matrix_attention import CoverageMatrixAttention
from summarize.modules.generate_probability_functions import GenerateProbabilityFunction
from summarize.modules.rnns import RNN
from summarize.nn.beam_search import BeamSearch
from summarize.nn.util import normalize_losses
from summarize.training.metrics import CrossEntropyMetric


@Model.register('sds-pointer-generator')
class PointerGeneratorModel(Model):
    """
    An implementation of a the Pointer-Generator model from See et al. (2017),
    "Get To The Point: Summarization with Pointer-Generator Networks"
    (https://arxiv.org/abs/1704.04368)

    Parameters
    ----------
    document_token_embedder: ``TextFieldEmbedder``
        The ``TextFieldEmbedder`` that will embed the document tokens.
    encoder: ``RNN``
        The RNN that will encode the sequence of document tokens.
    attention: ``CoverageMatrixAttention``
        The attention function that will be computed between the encoder and
        decoder hidden states.
    attention_layer: ``FeedForward``
        The ``attention_layer`` will be applied after the decoder hidden state
        and attention context are concatenated. The output should be the size
        of the decoder hidden state. This abstraction was created because sometimes
        a ``tanh`` unit is used after the projection and other times it is not.
        In our experience, this decision can make a big difference in terms of
        performance and training speed.
    decoder: ``RNN``
        The RNN that will produce the sequence of summary tokens.
    bridge: ``Bridge``, optional (default = ``None``)
        The bridge layer to use in between the encoder final state and the
        initial decoder hidden state. If ``None``, no layer will be used.
    generate_probability_function: ``GenerateProbabilityFunction``
        The function which will be used to compute p_gen.
    beam_search: ``BeamSearch``
        The ``BeamSearch`` object to use for prediction.
    run_beam_search: ``bool``
        Indicates whether or not beam search should be run during prediction. This
        is useful to turn off during training and on during testing if the
        beam search procedure is expensive.
    summary_token_embedder: ``TokenEmbedder``, optional (default = ``None``)
        The ``TokenEmbedder`` that will embed the summary tokens. If ``None``, the
        ``document_token_embedder``'s embedder for the ``"tokens"`` will be used.
    summary_namespace: ``str``, optional (default = ``"tokens"``)
        The namespace of the summary tokens which is used to map from the integer
        token representation to the string token representation.
    use_input_feeding: ``bool``, optional (default = ``False``)
        Indicates if input feeding should be used. See https://arxiv.org/pdf/1508.04025.pdf
        for details.
    input_feeding_projection_layer: ``FeedForward``, optional (default = ``None``)
        If input feeding is used, the ``input_feeding_projection_layer`` will optionally
        run on the concatenated input embedding and context vector. The output will
        be passed as input to the decoder. This is not specified in Luong et al. (2015),
        but it is used in See et al. (2017).
    instance_loss_normalization: ``str``
        The method for normalizing the loss per-instance. See `summarize.nn.util.normalize_losses`
        for more information.
    batch_loss_normalization: ``str``
        The method for normalizing the loss for the batch. See `summarize.nn.util.normalize_losses`
        for more information.
    coverage_loss_weight: ``float``, optional (default = ``1.0``)
        The weight to place on the coverage loss (lambda in See et al. (2017))
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
                 run_beam_search: bool = True,
                 summary_token_embedder: Optional[TokenEmbedder] = None,
                 summary_namespace: str = 'tokens',
                 use_input_feeding: bool = False,
                 input_feeding_projection_layer: Optional[FeedForward] = None,
                 instance_loss_normalization: str = 'sum',
                 batch_loss_normalization: str = 'average',
                 metrics: Optional[List[Metric]] = None,
                 coverage_loss_weight: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self.document_token_embedder = document_token_embedder
        self.encoder = encoder
        self.attention = attention
        self.attention_layer = attention_layer
        self.decoder = decoder
        self.bridge = bridge
        self.generate_probability_function = generate_probability_function
        self.beam_search = beam_search
        self.run_beam_search = run_beam_search
        self.summary_token_embedder = summary_token_embedder or document_token_embedder._token_embedders['tokens']
        self.summary_namespace = summary_namespace
        self.use_input_feeding = use_input_feeding
        self.input_feeding_projection_layer = input_feeding_projection_layer
        self.instance_loss_normalization = instance_loss_normalization
        self.batch_loss_normalization = batch_loss_normalization
        self.coverage_loss_weight = coverage_loss_weight
        # The ``output_layer`` is applied after the attention context and decoder
        # hidden state are combined. It is used to calculate the softmax over the
        # summary vocabulary
        self.output_layer = torch.nn.Linear(decoder.get_output_dim(), vocab.get_vocab_size(summary_namespace))

        # Retrieve some special vocabulary token indices. Some of them are
        # required to exist.
        token_to_index = vocab.get_token_to_index_vocabulary(summary_namespace)
        assert START_SYMBOL in token_to_index
        self.start_index = token_to_index[START_SYMBOL]
        assert END_SYMBOL in token_to_index
        self.end_index = token_to_index[END_SYMBOL]
        assert DEFAULT_PADDING_TOKEN in token_to_index
        self.pad_index = token_to_index[DEFAULT_PADDING_TOKEN]
        assert DEFAULT_OOV_TOKEN in token_to_index
        self.oov_index = token_to_index[DEFAULT_OOV_TOKEN]
        self.sent_start_index = None
        if SENT_START_SYMBOL in token_to_index:
            self.sent_start_index = token_to_index[SENT_START_SYMBOL]
        self.sent_end_index = None
        if SENT_END_SYMBOL in token_to_index:
            self.sent_end_index = token_to_index[SENT_END_SYMBOL]

        # Define the metrics that will be computed
        self.metrics = metrics
        self.cross_entropy_metric = CrossEntropyMetric()

        initializer(self)

    def _run_encoder(self, document: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Runs the encoder RNN over the document tokens and prepares the encoder's
        hidden states to be ready to initialize the decoder.

        Parameters
        ----------
        document: ``Dict[str, torch.Tensor]``
            The document tokens.

        Returns
        -------
        encoder_outputs: ``torch.Tensor``, ``(batch_size, num_document_tokens, encoder_hidden_size)``
            The hidden state outputs from the encoder.
        document_mask: ``torch.Tensor``, ``(batch_size, num_document_tokens)``
            The document tokens mask.
        hidden: ``torch.Tensor``, ``(batch_size, decoder_hidden_size)``
            The hidden state that should be used to initialize the decoder
        memory: ``torch.Tensor``, ``(batch_size, decoder_hidden_size)``
            The memory state that should be used to initialize the decoder
        """
        # Encoder the document tokens
        # shape: (batch_size, num_document_tokens)
        document_mask = get_text_field_mask(document)
        # shape: (batch_size, num_document_tokens, embedding_size)
        document_token_embeddings = self.document_token_embedder(document)
        # shape: (batch_size, num_document_tokens, encoder_hidden_size)
        # shape: (num_layers * num_directions, batch_size, encoder_hidden_size)
        encoder_outputs, hidden = self.encoder(document_token_embeddings, document_mask)

        # Reshape the encoder's hidden state(s) for decoding
        # shape: (num_layers, batch_size, encoder_hidden_size * num_directions)
        hidden = self.encoder.reshape_hidden_for_decoder(hidden)
        # For now, we only support ``num_layers = 1``. The beam search logic
        # requires handling tensors where the first dimension is the batch size.
        # Implementing a larger number of layers would require messing with the
        # dimensions, and we haven't put the effort in to do that.
        message = f'Currently, only ``num_layers = 1`` is supported.'
        if isinstance(hidden, tuple):
            if hidden[0].size(0) != 1:
                raise Exception(message)
            # shape: (batch_size, encoder_hidden_size * num_directions)
            hidden = hidden[0].squeeze(0), hidden[1].squeeze(0)
        else:
            if hidden.size(0) != 1:
                raise Exception(message)
            # shape: (batch_size, encoder_hidden_size * num_directions)
            hidden = hidden.squeeze(0)

        # Apply the bridge layer
        if self.bridge is not None:
            # shape: (batch_size, decoder_hidden_size)
            hidden = self.bridge(hidden)

        # Split the hidden state's tuple items for decoding purposes. The generic
        # beam search code expects tensors as values in the state dictionary, so
        # we can't use the default tuple-based implementation. This means we have
        # to create a ``memory`` tensor even if it's not used (e.g., by a GRU) or
        # else the reshaping logic of the decoding will fail. However, it will not
        # be used.
        if self.encoder.has_memory():
            # shape: (batch_size, encoder_hidden_size * num_directions)
            hidden, memory = hidden
        else:
            memory = hidden.new_zeros(hidden.size())

        return encoder_outputs, document_mask, hidden, memory

    def _compute_loss(self,
                      initial_decoding_state: Dict[str, torch.Tensor],
                      summary: Dict[str, torch.Tensor],
                      summary_token_document_indices: torch.Tensor,
                      summary_token_document_indices_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes the training loss, which is the log of the vocabulary
        and copy probabilities weighted by the generate probability.

        Parameters
        ----------
        initial_decoding_state: ``Dict[str, torch.Tensor]``
            The dictionary with the tensors used to initialize decoding.
        summary: ``Dict[str, torch.Tensor]``
            The summary tokens.
        summary_token_document_indices: (batch_size, num_summary_tokens, num_matches)
            The index of each summary token in the document
        summary_token_document_indices_mask: (batch_size, num_summary_tokens, num_matches)
            The mask for ``summary_token_document_indices`` to mark the valid indices

        Returns
        -------
        torch.Tensor: (1)
            The loss
        torch.Tensor: (1)
            The token-level cross-entropy
        """
        # Get the summary tokens from the dictionary
        # shape: (batch_size, num_summary_tokens)
        summary_tokens = summary['tokens']
        batch_size, num_summary_tokens = summary_tokens.size()

        # The tokens that we feed into the decoder are all but the last time
        # step, which has the <eos> token.
        # shape: (batch_size, num_summary_tokens - 1)
        summary_input_tokens = summary_tokens[:, :-1]

        # The target tokens are from the first time step onward
        # shape: (batch_size, num_summary_tokens - 1)
        summary_target_tokens = summary_tokens[:, 1:].contiguous()
        # Similarly, we need to know the location of each target token in the
        # document, so strip off the first index
        # shape: (batch_size, num_summary_tokens - 1, num_matches)
        summary_token_document_indices = summary_token_document_indices[:, 1:]
        # shape: (batch_size, num_summary_tokens - 1, num_matches)
        summary_token_document_indices_mask = summary_token_document_indices_mask[:, 1:]

        # Pass the input tokens through the decoding step
        # shape: (batch_size, num_summary_tokens - 1, summary_vocab_size)
        logits, state = self._decoder_step(summary_input_tokens, initial_decoding_state)

        # Extract the information from the state necessary to compute the loss
        # shape: (batch_size, num_summary_tokens - 1)
        p_gen = state['p_gen']
        # shape: (batch_size, num_summary_tokens - 1, num_document_tokens)
        attention_probabilities = state['attention']
        # shape: (batch_size, num_summary_tokens - 1, num_document_tokens)
        coverage_vectors = state['coverage_vectors']
        # shape: (batch_size, num_document_tokens)
        document_mask = state['document_mask']

        # Compute the two different probabilities
        # shape: (batch_size, num_summary_tokens - 1)
        vocab_log_probs = self._compute_vocab_log_probs(logits, summary_target_tokens)

        # shape: (batch_size, num_summary_tokens - 1)
        copy_log_probs = self._compute_copy_log_probs(attention_probabilities,
                                                      summary_token_document_indices,
                                                      summary_token_document_indices_mask)

        # shape: (batch_size, num_summary_tokens - 1)
        log_p_gen = torch.log(p_gen)
        # shape: (batch_size, num_summary_tokens - 1)
        log_p_copy = torch.log(1.0 - p_gen)
        # shape: (batch_size, num_summary_tokens - 1)
        combined_log_probs = torch.stack([log_p_gen + vocab_log_probs,
                                          log_p_copy + copy_log_probs], dim=2)
        # shape: (batch_size, num_summary_tokens - 1)
        nll_losses = -torch.logsumexp(combined_log_probs, dim=2)

        if self.coverage_loss_weight > 0.0:
            # shape: (batch_size, num_summary_tokens - 1)
            coverage_losses = self._compute_coverage_loss(attention_probabilities, coverage_vectors,
                                                          document_mask)
            # shape: (batch_size, num_summary_tokens - 1)
            losses = nll_losses + self.coverage_loss_weight * coverage_losses
        else:
            losses = nll_losses

        losses_mask = (summary_target_tokens != self.pad_index).float()
        loss = normalize_losses(losses, losses_mask,
                                self.instance_loss_normalization,
                                self.batch_loss_normalization)

        # Compute the token-level cross-entropy
        total_loss = (nll_losses * losses_mask).sum()
        num_targets = losses_mask.sum()
        self.cross_entropy_metric(total_loss.item(), num_targets.item())

        return loss

    def _compute_vocab_log_probs(self,
                                 logits: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probability of generating the target tokens under the model.

        Parameters
        ----------
        logits: (batch_size, num_targets, vocab_size)
            The raw vocabulary scores
        targets: (batch_size, num_targets)
            The ground-truth tokens

        Returns
        -------
        torch.Tensor: (batch_size, num_targets)
            The loss for each individual token.
        """
        batch_size, num_targets = targets.size()
        # shape: (batch_size, num_targets, vocab_size)
        log_distribution = F.log_softmax(logits, dim=2)
        # shape: (batch_size, num_targets)
        log_probs = torch.gather(log_distribution, 2, targets.unsqueeze(2)).squeeze(2)
        return log_probs

    def _compute_copy_log_probs(self,
                                attention_probabilities: torch.Tensor,
                                summary_token_document_indices: torch.Tensor,
                                summary_token_document_indices_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probability of copying the target token under the model.

        Parameters
        ----------
        attention_probabilities: (batch_size, num_targets, num_document_tokens)
            The attention distribution for each time step
        summary_token_document_indices: (batch_size, num_targets, num_matches)
            The index of each summary token in the document
        summary_token_document_indices_mask: (batch_size, num_targets, num_matches)
            The mask for ``summary_token_document_indices`` to mark the valid indices

        Returns
        -------
        torch.Tensor: (batch_size, num_targets)
            The loss for each individual token.
        """
        # Gather all of the probabilities for the target tokens
        # shape: (batch_size, num_targets, num_matches)
        copy_probabilities = torch.gather(attention_probabilities, 2, summary_token_document_indices)
        # shape: (batch_size, num_targets, num_matches)
        copy_probabilities = copy_probabilities * summary_token_document_indices_mask
        # shape: (batch_size, num_targets)
        copy_probabilities = copy_probabilities.sum(dim=2)

        # Some of the tokens will not appear in the document, which means
        # they will have 0 probability here. This will cause numerical problems,
        # so we add a small value to the probabilities to prevent that
        # shape: (batch_size, num_targets)
        copy_probabilities = copy_probabilities + 1e-20

        # shape: (batch_size, num_targets)
        log_copy_probabilities = torch.log(copy_probabilities)
        return log_copy_probabilities

    def _compute_coverage_loss(self,
                               attention_probabilities: torch.Tensor,
                               coverage_vectors: torch.Tensor,
                               document_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the coverage loss for each step of decoding. The method does
        not multiply the loss by any weight hyperparameters.

        Parameters
        ----------
        attention_probabilities: (batch_size, num_summary_tokens, num_document_tokens)
            The attention probabilities
        coverage_vectors: (batch_size, num_summary_tokens, num_document_tokens)
            The accumulated attention probability distributions over every decoding step
        document_mask: (batch_size, num_document_tokens)
            The document token mask

        Returns
        -------
        torch.Tensor: (batch_size, num_summary_tokens)
            The coverage loss for each decoding step
        """
        # shape: (batch_size, num_summary_tokens, num_document_tokens)
        losses = torch.min(attention_probabilities, coverage_vectors)
        # Mask away any loss attributed to the masked document tokens. This
        # step is redundant because the attention probabilities for these tokens
        # should be 0.
        # shape: (batch_size, num_summary_tokens, num_document_tokens)
        mask = document_mask.unsqueeze(1).expand_as(losses)
        # shape: (batch_size, num_summary_tokens, num_document_tokens)
        losses = losses * mask.float()
        # shape: (batch_size, num_summary_tokens)
        losses = losses.sum(dim=2)
        return losses

    def _decoder_step(self,
                      summary_tokens: torch.Tensor,
                      state: Dict[str, torch.Tensor],
                      token_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Runs the decoder one step for every input token. This function implements
        the interface for AllenNLP's generic beam search code. Instead of a ``batch_size``,
        this function uses a ``group_size`` since the beam search will run in
        parallel for each batch.

        Parameters
        ----------
        summary_tokens: ``torch.Tensor``, ``(group_size, num_summary_tokens)`` or ``(group_size,)
            The tokens which should be input to the next step. The decoder will
            run one time for each token time step. If there is only one dimension, the function
            is being called during inference and will only run 1 decoder step.
        state: ``Dict[str, torch.Tensor]``
            The current decoder state.
        token_mask: ``torch.Tensor``, (group_size, num_summary_tokens)
            An optional mask to apply during the encoding

        Returns
        -------
        logits: ``torch.Tensor``, ``(group_size, num_summary_tokens)`` or ``(batch_size,)``
            The vocabulary scores for each time step. If this is being used during
            inference, the one-dimensional tensor will be returned.
        state: ``Dict[str, torch.Tensor]``
            The updated decoder state.
        """
        # Since this method is written generically for processing multiple time steps
        # in the same call and for implementing the AllenNLP beam search interface,
        # we add a dimension to the input tokens if there is no dimension for the time step.
        is_inference = summary_tokens.dim() == 1
        if is_inference:
            # shape: (group_size, 1)
            summary_tokens = summary_tokens.unsqueeze(-1)
        group_size, num_summary_tokens = summary_tokens.size()

        # Unpack everything from the input state
        # shape: (group_size, num_document_tokens)
        document_mask = state['document_mask']
        # shape: (group_size, num_document_tokens)
        document_token_first_indices = state['document_token_first_indices']
        # shape: (group_size, num_document_tokens)
        document_in_summary_namespace = state['document_in_summary_namespace']
        # shape: (group_size, num_document_tokens, encoder_hidden_size)
        encoder_outputs = state['encoder_outputs']
        # shape: (group_size, decoder_hidden_size)
        hidden = state['hidden'].contiguous()
        # shape: (group_size, decoder_hidden_size)
        memory = state['memory'].contiguous()
        # shape: (group_size, decoder_hidden_size)
        input_feed = state['input_feed']
        # shape: (group_size, num_document_tokens)
        coverage = state['coverage']

        # Get the token embeddings
        # shape: (group_size, num_summary_tokens, embedding_size)
        summary_tokens = self._get_input_token(summary_tokens)
        # shape: (group_size, num_summary_tokens, embedding_size)
        summary_token_embeddings = self.summary_token_embedder(summary_tokens)

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

        # If we use input feeding, we have to manually pass all of the summary
        # tokens through the decoder with a for loop. Otherwise, we can use
        # the vectorized version, which should be faster
        if self.use_input_feeding:
            original_decoder_outputs_list = []
            decoder_outputs = []
            attention_probabilities_list = []
            attention_contexts_list = []
            input_embeddings = []
            coverage_vectors_list = []
            for i in range(num_summary_tokens):
                # Setup the input to the decoder, optionally using input feeding
                # shape: (batch_size, embedding_size)
                input_embedding = summary_token_embeddings[:, i, :]
                # shape: (batch_size, input_feeding_size)
                input_embedding = self._apply_input_feeding(input_embedding, input_feed)
                # shape: (batch_size, 1, embedding_size)
                input_embedding = input_embedding.unsqueeze(1)

                # Pass the input through the decoder
                # shape: (group_size, 1, decoder_hidden_size)
                # shape: (group_size, 1, decoder_hidden_size)
                # shape: (1, group_size, decoder_hidden_size)
                # shape: (group_size, 1, num_document_tokens)
                # shape: (group_size, 1, encoder_hidden_size)
                # shape: (group_size, 1, num_documents_tokens)
                # shape: (group_size, num_documents_tokens)
                (original_decoder_outputs, decoder_output, decoder_state, attention_probabilities, attention_context,
                    coverage_vectors, coverage) = \
                    self._decoder_forward(input_embedding, decoder_state, encoder_outputs, document_mask, coverage, None)

                original_decoder_outputs_list.append(original_decoder_outputs)
                decoder_outputs.append(decoder_output)
                attention_probabilities_list.append(attention_probabilities)
                attention_contexts_list.append(attention_context)
                input_embeddings.append(input_embedding)
                coverage_vectors_list.append(coverage_vectors)
                # Take the new ``input_feed`` vector
                # shape: (group_size, decoder_hidden_size)
                input_feed = decoder_output.squeeze(1)

            # Combine all the decoder outputs and attention probabilities
            # shape: (group_size, num_summary_tokens, decoder_hidden_size)
            original_decoder_outputs = torch.cat(original_decoder_outputs_list, dim=1)
            # shape: (group_size, num_summary_tokens, decoder_hidden_size)
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            # shape: (group_size, num_summary_tokens, num_document_tokens)
            attention_probabilities = torch.cat(attention_probabilities_list, dim=1)
            # shape: (group_size, num_summary_tokens, encoder_hidden_size)
            attention_contexts = torch.cat(attention_contexts_list, dim=1)
            # shape: (group_size, num_summary_tokens, embedding_size)
            input_embeddings = torch.cat(input_embeddings, dim=1)
            # shape: (group_size, num_summary_tokens, num_documents_tokens)
            coverage_vectors = torch.cat(coverage_vectors_list, dim=1)
        else:
            # shape: (group_size, num_summary_tokens, embedding_size)
            input_embeddings = summary_token_embeddings
            # shape: (group_size, num_summary_tokens, decoder_hidden_size)
            # shape: (group_size, num_summary_tokens, decoder_hidden_size)
            # shape: (1, group_size, decoder_hidden_size)
            # shape: (group-size, num_summary_tokens, num_document_tokens)
            # shape: (group-size, num_summary_tokens, encoder_hidden_size)
            # shape: (group-size, num_summary_tokens, num_document_tokens)
            # shape: (group-size, num_document_tokens)
            (original_decoder_outputs, decoder_outputs, decoder_state, attention_probabilities, attention_contexts,
                coverage_vectors, coverage) = \
                self._decoder_forward(input_embeddings, decoder_state, encoder_outputs, document_mask, coverage, token_mask)

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

        # Project the hidden state to get a score for each vocabulary token
        # shape: (group_size, num_summary_tokens, summary_vocab_size)
        logits = self.output_layer(decoder_outputs)

        # Compute the soft switch that decides whether to copy or generate
        # shape: (batch_size, num_summary_tokens)
        p_gen = self.generate_probability_function(input_embeddings, original_decoder_outputs,
                                                   decoder_outputs, attention_contexts)

        # At this point in the code, the logic switches depending on if
        # we are computing the loss or doing inference because we can efficiently
        # compute the loss more efficiently than calculating the full softmax
        # over the vocabulary and document tokens
        if is_inference:
            # We know there is only one summary token, so we squeeze the dimension
            # to make computation a little easier
            # shape: (group_size, summary_vocab_size)
            logits = logits.squeeze(1)
            # shape: (group_size, num_document_tokens)
            attention_probabilities = attention_probabilities.squeeze(1)
            # shape: (batch_size)
            p_gen = p_gen.squeeze(1)

            # shape: (batch_size, summary_vocab_size)
            output_scores = self._compute_full_log_softmax(logits, attention_probabilities,
                                                           p_gen, document_token_first_indices,
                                                           document_in_summary_namespace)
        else:
            output_scores = logits

        # Update the state dictionary
        output_state = dict(state)
        output_state['hidden'] = hidden
        output_state['memory'] = memory
        output_state['input_feed'] = input_feed
        output_state['attention'] = attention_probabilities
        output_state['coverage'] = coverage
        output_state['p_gen'] = p_gen
        output_state['coverage_vectors'] = coverage_vectors

        return output_scores, output_state

    def _get_input_token(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Potentially replaces the index in the ``tokens`` tensor with the
        index of the copy token if the index was copied. This will happen during
        inference when the highest probability item comes from the document, and
        this is represented with an index that is offset by the vocabulary size.

        Parameters
        ----------
        tokens:
            The input tokens

        Returns
        -------
        The input tokens with some replaced with the copy token.
        """
        vocab_size = self.vocab.get_vocab_size(self.summary_namespace)
        copy_mask = (tokens >= vocab_size).long()
        return (1 - copy_mask) * tokens + copy_mask * self.oov_index

    def _apply_input_feeding(self,
                             embedding: torch.Tensor,
                             input_feed: torch.Tensor) -> torch.Tensor:
        """
        Applies input feeding to combine the embedding and context vectors.

        Parameters
        ----------
        embedding: ``torch.Tensor``, ``(batch_size, embedding_size)``
            The summary token embeddings
        input_feed: ``torch.Tensor``, ``(batch_size, encoder_hidden_size)``
            The input feeding vector

        Returns
        -------
        embedding: ``torch.Tensor``, ``(batch_size, input_feeding_size)``
            The combined vector which should be passed as input the decoder.
        """
        # shape: (batch_size, embedding_size + decoder_hidden_size)
        input_embedding = torch.cat([embedding, input_feed], dim=1)
        if self.input_feeding_projection_layer is not None:
            # shape: (batch_size, input_feeding_size)
            input_embedding = self.input_feeding_projection_layer(input_embedding)
        return input_embedding

    def _decoder_forward(self,
                         input_vectors: torch.Tensor,
                         hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                         encoder_outputs: torch.Tensor,
                         encoder_mask: torch.Tensor,
                         coverage: torch.Tensor,
                         input_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the RNN decoder on the input vectors and hidden state and applies
        the attention mechanism.

        Parameters
        ----------
        input_vectors: ``torch.Tensor``, ``(batch_size, num_summary_tokens, input_size)``
            The input vectors.
        hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]``
            The RNN hidden state.
        encoder_outputs: ``torch.Tensor``, ``(batch_size, num_document_tokens, encoder_hidden_size)``
            The encoder output states.
        encoder_mask: ``torch.Tensor``, ``(batch_size, num_document_tokens)``
            The document mask.
        coverage: ``torch.Tensor``, ``(batch_size, num_document_tokens)``
            The coverage vector
        input_mask: ``torch.Tensor``, ``(batch_size, num_summary_tokens)``
            An optional mask for the summary tokens.

        Returns
        -------
        original_decoder_outputs: ``torch.Tensor``, ``(batch_size, num_summary_tokens, decoder_hidden_size)``
            The decoder hidden representation without attention.
        decoder_outputs: ``torch.Tensor``, ``(batch_size, num_summary_tokens, decoder_hidden_size)``
            The decoder hidden representation with attention.
        hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]``
            The final hidden states.
        attention_probabilities: ``torch.Tensor``, ``(batch_size, num_summary_tokens, num_document_tokens)``
            The attention probabilities over the document tokens for each summary token
        attention_context: ``torch.Tensor``, ``(batch_size, num_summary_tokens, encoder_hidden_size)``
            The context vector for each of the summary tokens
        coverage_vectors: ``torch.Tensor``, ``(batch_size, num_summary_tokens, encoder_hidden_size)``
            The coverage vectors which were used to compute the attention probabilities
        last_coverage_vector: ``torch.Tensor``, ``(batch_size, encoder_hidden_size)``
            The coverage vector after computing all of the attention probabilities
        """
        # shape: (group_size, num_summary_tokens, decoder_hidden_size)
        # shape: (1, group_size, decoder_hidden_size)
        decoder_outputs, hidden = self.decoder(input_vectors, input_mask, hidden)

        # Incorporate attention
        # shape: (group_size, num_summary_tokens, num_document_tokens)
        # shape: (group_size, num_summary_tokens, num_document_tokens)
        # shape: (group_size, num_document_tokens)
        attention_probabilities, coverage_vectors, last_coverage_vector = \
            self.attention(decoder_outputs, encoder_outputs, encoder_mask, coverage)
        # shape: (group_size, num_summary_tokens, encoder_hidden_size)
        attention_context = weighted_sum(encoder_outputs, attention_probabilities)

        # Concatenate the attention context with the decoder outputs
        # then project back to the decoder hidden size
        # shape: (group_size, num_summary_tokens, encoder_hidden_size + decoder_hidden_size)
        concat = torch.cat([attention_context, decoder_outputs], dim=2)

        # shape: (group_size, num_summary_tokens, decoder_hidden_size)
        projected_hidden = self.attention_layer(concat)

        return decoder_outputs, projected_hidden, hidden, attention_probabilities, attention_context, \
            coverage_vectors, last_coverage_vector

    def _compute_full_log_softmax(self,
                                  logits: torch.Tensor,
                                  attention_probabilities: torch.Tensor,
                                  p_gen: torch.Tensor,
                                  document_token_first_indices: torch.Tensor,
                                  document_in_summary_namespace: torch.Tensor) -> None:
        """
        Computes the full log-probability distribution over all of the tokens
        in the vocabulary and in the document. The probability mass for tokens
        in the document is aggregated on the first instance of that token in
        the document if it does not exist in the summary vocabulary. Otherwise,
        the mass will be added to the summary vocabulary index. The vocabulary
        tokens are the first indices in the output tensor, followed by the
        document tokens.

        Parameters
        ----------
        logits: (batch_size, vocab_size)
            The logit scores of the generation distribution
        attention_probabilities: (batch_size, num_document_tokens)
            The attention distribution for each decoding time step
        p_gen: (batch_size)
            The value of the soft switch between generating and copying.
        document_token_first_indices: (batch_size, num_document_tokens)
            The first location in the document of the tokens.
        document_in_summary_namespace: (batch_size, num_document_tokens)
            The document tokens but represented in the summary namespace.

        Returns
        -------
        torch.Tensor: (batch_size, vocab_size + num_document_tokens)
            The log-probabilities
        """
        batch_size, vocab_size = logits.size()
        num_document_tokens = attention_probabilities.size(1)

        # shape: (batch_size)
        log_p_gen = torch.log(p_gen)
        # shape: (batch_size)
        log_p_copy = torch.log(1.0 - p_gen)

        # Copy the logits into an expanded tensor with epsilon scores for
        # the document tokens so there aren't numerical issues
        # shape: (batch_size, num_document_tokens)
        epsilon_tensor = logits.new_full((batch_size, num_document_tokens), fill_value=1e-30)
        # shape: (batch_size, vocab_size + num_document_tokens)
        expanded_logits = torch.cat([logits, epsilon_tensor], dim=1)
        # shape: (batch_size, vocab_size + num_document_tokens)
        vocab_log_probs = torch.log_softmax(expanded_logits, dim=1) + log_p_gen.unsqueeze(1).expand_as(expanded_logits)

        # Iterate over the attention distribution and aggregate
        # the probabilities of multiple occurences of the tokens
        # shape: (batch_size, vocab_size + num_document_tokens)
        copy_probs = vocab_log_probs.new_zeros(batch_size, vocab_size + num_document_tokens)

        for i in range(num_document_tokens):
            # Determine what index the probability mass for this document token
            # should be added to. If the token is in the summary vocabulary, it
            # should be added to the vocabulary index. Otherwise, it should be
            # added to the first position that token appears in the document.
            # shape: (batch_size, 1)
            tokens = document_in_summary_namespace[:, i].unsqueeze(1)
            # shape: (batch_size, 1)
            first_index = document_token_first_indices[:, i].unsqueeze(1)
            # shape: (batch_size, 1)
            oov_mask = (tokens == self.oov_index).long()
            # shape: (batch_size, 1)
            target_index = (1 - oov_mask) * tokens + oov_mask * (first_index + vocab_size)

            # Aggregate the probabilities in the correct indices
            # shape: (batch_size, 1)
            probabilities = attention_probabilities[:, i].unsqueeze(1)
            copy_probs.scatter_add_(1, target_index, probabilities)

        # shape: (batch_size, vocab_size + num_document_tokens)
        copy_log_probs = torch.log(copy_probs) + log_p_copy.unsqueeze(1).expand_as(copy_probs)

        # shape: (batch_size, vocab_size + num_document_tokens, 2)
        combined_log_probs = torch.stack([vocab_log_probs, copy_log_probs], dim=2)
        # shape: (batch_size, vocab_size + num_document_tokens)
        log_probs = torch.logsumexp(combined_log_probs, dim=2)
        return log_probs

    def _run_inference(self,
                       initial_decoding_state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference given the initial decoder state. Beam search is
        implemented using AllenNLP's generic beam search logic.

        Parameters
        ----------
        initial_decoding_state: ``Dict[str, torch.Tensor]``
            The initial decoding state.

        Returns
        -------
        predictions: ``torch.Tensor``, ``(batch_size, beam_size, max_output_length)``
            The beam_size'd predictions for each batch.
        log_probabilities: ``torch.Tensor``, ``(batch_size, beam_size)``
            The log-probabilities for each output sequence.
        """
        # Pull out a tensor to get the device and batch_size
        hidden = initial_decoding_state['hidden']
        batch_size = hidden.size(0)

        initial_predictions = hidden.new_empty(batch_size, dtype=torch.long)
        initial_predictions.fill_(self.start_index)

        # shape: (batch_size, beam_size, max_output_length)
        # shape: (batch_size, beam_size)
        predictions, log_probabilities = \
            self.beam_search.search(initial_predictions, initial_decoding_state, self._decoder_step)
        return predictions, log_probabilities

    def _convert_indices_to_string(self,
                                   indices: torch.Tensor,
                                   document_tokens: List[List[str]]) -> List[str]:
        """
        Converts a tensor of token indices to string representations. This is
        used during prediction to convert the output to normal tokens. The document
        tokens are required because if the index represents copying a token
        from the document, the corresponding token is used.

        Parameters
        ----------
        indices: (batch_size, num_tokens)
            The tokens to convert to strings
        document_tokens:
            The batched document tokens.

        Returns
        -------
        ``List[str]``:
            The tokens joined into summaries
        """
        vocab_size = self.vocab.get_vocab_size(self.summary_namespace)
        summaries = []
        for batch in range(len(indices)):
            tokens = []
            for i in range(len(indices[batch])):
                # The best prediction is in the 0th beam position
                index = indices[batch][i].item()
                # We skip the start, sentence start, and sentence end indices. It is ok
                # if these are ``None`` because the index should never be ``None``
                if index in [self.start_index, self.sent_start_index, self.sent_end_index]:
                    continue
                # We stop decoding if we see the end or pad symbols
                if index in [self.end_index, self.pad_index]:
                    break
                # If the token is larger than or equal to the vocabulary size, it was copied
                # from the document
                if index >= vocab_size:
                    index = index - vocab_size
                    token = document_tokens[batch][index]
                else:
                    token = self.vocab.get_token_from_index(index, self.summary_namespace)
                tokens.append(token)
            summary = ' '.join(tokens)
            summaries.append(summary)
        return summaries

    def _update_metrics(self,
                        predictions: torch.Tensor,
                        metadata: List[Dict[str, Any]],
                        summary_field_name: str = 'summary') -> None:
        """
        Updates the metrics based on the predictions and ground-truth summaries.

        Parameters
        ----------
        predictions: ``torch.Tensor``, ``(batch_size, beam_size, max_output_length)``
            The output predictions from beam search.
        metadata: ``List[Dict[str, Any]]``
            The batched metadata. In order to successfully update the metrics, the
            ``"summary"`` must be a key in the metadata.
        """
        # If we have acess to the ground-truth summaries, then we can
        # compute metrics
        batch_size = len(predictions)
        if summary_field_name in metadata[0] and self.metrics is not None:
            gold_summaries = [metadata[batch][summary_field_name] for batch in range(batch_size)]
            # shape: (batch_size, max_output_length)
            model_summaries = [prediction[0] for prediction in predictions]
            document_tokens = [metadata[batch]['document_tokens'] for batch in range(batch_size)]
            model_summaries = self._convert_indices_to_string(model_summaries, document_tokens)
            for metric in self.metrics:
                metric(gold_summaries=gold_summaries, model_summaries=model_summaries)

    @overrides
    def forward(self,
                document: Dict[str, torch.Tensor],
                document_token_first_indices: torch.Tensor,
                document_in_summary_namespace: torch.Tensor,
                metadata: List[Dict[str, Any]],
                summary: Optional[Dict[str, torch.Tensor]] = None,
                summary_token_document_indices: Optional[torch.Tensor] = None,
                summary_token_document_indices_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Computes the forward pass for the ``PointerGeneratorModel``.

        Parameters
        ----------
        document: ``Dict[str, torch.Tensor]``
            The document tokens.
        document_token_first_indices: (batch_size, num_document_tokens)
            The first index in the document where each of the document tokens occurs
        document_in_summary_namespace: (batch_size, num_document_tokens)
            The index of every token in the document but in the summary namespace.
        metadata: ``List[Dict[str, Any]]``
            The metadata for this batch.
        summary: ``Dict[str, torch.Tensor]``, optional (default = ``None``)
            The summary tokens. The dictionary must have some representation for the
            summary using the "tokens" key since it is used during training.
            If ``None``, the loss will not be calculated.
        summary_token_document_indices: (batch_size, num_summary_tokens, num_matches)
            The index of every location in the document that each summary token
            appears.
        summary_token_document_indices_mask: (batch_size, num_summary_tokens, num_matches)
            The mask for ``summary_token_document_indices`` which indicates which
            values are valid.
        """
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
        document_in_summary_namespace = document_in_summary_namespace.long()

        # Setup the state which will be used to initialize decoding
        initial_decoding_state = {
            'document_mask': document_mask,
            'encoder_outputs': encoder_outputs,
            'hidden': hidden,
            'memory': memory,
            'input_feed': input_feed,
            'coverage': coverage,
            'document_token_first_indices': document_token_first_indices,
            'document_in_summary_namespace': document_in_summary_namespace
        }

        output_dict = {}

        # Compute the loss if we have the ground-truth summaries
        if summary is not None:
            summary_token_document_indices = summary_token_document_indices.long()

            output_dict['loss'] = self._compute_loss(initial_decoding_state, summary,
                                                     summary_token_document_indices,
                                                     summary_token_document_indices_mask)

        # If we aren't training, then we need to do inference
        if not self.training and self.run_beam_search:
            # shape: (batch_size, beam_size, max_output_length)
            # shape: (batch_size, beam_size)
            predictions, log_probabilities = self._run_inference(initial_decoding_state)
            output_dict['predictions'] = predictions
            output_dict['log_probabilities'] = log_probabilities
            self._update_metrics(predictions, metadata)

        # Always include the metadata in the output dictionary
        output_dict['metadata'] = metadata
        return output_dict

    @overrides
    def decode(self,
               output_dict: Dict[str, torch.Tensor],
               summary_field_name: str = 'summary') -> Dict[str, Any]:
        predictions = output_dict.pop('predictions')
        metadata = output_dict.pop('metadata')
        batch_size = len(predictions)

        top_predictions = [prediction[0] for prediction in predictions]
        document_tokens = [metadata[batch]['document_tokens'] for batch in range(batch_size)]
        summaries = self._convert_indices_to_string(top_predictions, document_tokens)

        output_dict[summary_field_name] = summaries
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get_metric(reset))
        metrics.update(self.cross_entropy_metric.get_metric(reset))
        return metrics
