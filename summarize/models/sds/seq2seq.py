import torch
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, MatrixAttention, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Any, Dict, List, Optional, Tuple, Union

from summarize.common.util import SENT_START_SYMBOL, SENT_END_SYMBOL
from summarize.modules.bridge import Bridge
from summarize.modules.rnns import RNN
from summarize.nn.beam_search import BeamSearch
from summarize.nn.util import normalize_losses
from summarize.training.metrics import CrossEntropyMetric


@Model.register('sds-seq2seq')
class Seq2SeqModel(Model):
    """
    An implementation of a standard encoder-decoder network with attention
    built on RNNs.

    Parameters
    ----------
    document_token_embedder: ``TextFieldEmbedder``
        The ``TextFieldEmbedder`` that will embed the document tokens.
    encoder: ``RNN``
        The RNN that will encode the sequence of document tokens.
    attention: ``MatrixAttention``
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
    beam_search: ``BeamSearch``
        The ``BeamSearch`` object to use for prediction and validation.
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
                 run_beam_search: bool = True,
                 summary_token_embedder: Optional[TokenEmbedder] = None,
                 summary_namespace: str = 'tokens',
                 use_input_feeding: bool = False,
                 input_feeding_projection_layer: Optional[FeedForward] = None,
                 instance_loss_normalization: str = 'sum',
                 batch_loss_normalization: str = 'average',
                 metrics: Optional[List[Metric]] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self.document_token_embedder = document_token_embedder
        self.encoder = encoder
        self.attention = attention
        self.attention_layer = attention_layer
        self.decoder = decoder
        self.bridge = bridge
        self.beam_search = beam_search
        self.run_beam_search = run_beam_search
        self.summary_token_embedder = summary_token_embedder or document_token_embedder._token_embedders['tokens']
        self.summary_namespace = summary_namespace
        self.use_input_feeding = use_input_feeding
        self.input_feeding_projection_layer = input_feeding_projection_layer
        self.instance_loss_normalization = instance_loss_normalization
        self.batch_loss_normalization = batch_loss_normalization
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
        self.sent_start_index = None
        if SENT_START_SYMBOL in token_to_index:
            self.sent_start_index = token_to_index[SENT_START_SYMBOL]
        self.sent_end_index = None
        if SENT_END_SYMBOL in token_to_index:
            self.sent_end_index = token_to_index[SENT_END_SYMBOL]

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_index, reduction='none')

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

    def _run_teacher_forcing(self,
                             initial_decoding_state: Dict[str, torch.Tensor],
                             summary: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Runs teacher forcing and computes the scores over the vocabulary for
        every decoding timestep.

        Parameters
        ----------
        initial_decoding_state: ``Dict[str, torch.Tensor]``
            The dictionary with the tensors used to initialize decoding.
        summary: ``Dict[str, torch.Tensor]``
            The summary tokens.

        Returns
        ------
        logits: ``torch.Tensor``, ``(batch_size, num_summary_tokens - 1, vocab_size)``
            The unnormalized scores over the vocabulary for each time step.
        targets: ``torch.Tensor``, ``(batch_size, num_summary_tokens - 1, vocab_size)``
            The ground-truth target summary tokens that should be used to compute the loss.
        """
        # Get the summary tokens from the dictionary
        # shape: (batch_size, num_summary_tokens)
        summary_tokens = summary['tokens']

        # The tokens that we feed into the decoder are all but the last time
        # step, which has the <eos> token.
        # shape: (batch_size, num_summary_tokens - 1)
        summary_input_tokens = summary_tokens[:, :-1]

        # The target tokens are from the first time step onward
        # shape: (batch_size, num_summary_tokens - 1)
        summary_target_tokens = summary_tokens[:, 1:].contiguous()

        # Pass the input tokens through the decoding step
        # shape: (batch_size, num_summary_tokens - 1, summary_vocab_size)
        logits, _ = self._decoder_step(summary_input_tokens, initial_decoding_state)
        return logits, summary_target_tokens

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

        # Unpack everything from the input state. The attention probabilities
        # are there, but we don't need them for computation. They are only for
        # the beam search.
        # shape: (group_size, num_document_tokens)
        document_mask = state['document_mask']
        # shape: (group_size, num_document_tokens, encoder_hidden_size)
        encoder_outputs = state['encoder_outputs']
        # shape: (group_size, decoder_hidden_size)
        hidden = state['hidden']
        # shape: (group_size, decoder_hidden_size)
        memory = state['memory']
        # shape: (group_size, decoder_hidden_size)
        input_feed = state['input_feed']

        # Get the token embeddings
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
            decoder_outputs = []
            attention_probabilities_list = []
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
                # shape: (1, group_size, decoder_hidden_size)
                # shape: (group_size, 1, num_document_tokens)
                decoder_output, decoder_state, attention_probabilities = \
                    self._decoder_forward(input_embedding, decoder_state, encoder_outputs, document_mask, None)

                # Save the decoder output and attention probabilities
                decoder_outputs.append(decoder_output)
                attention_probabilities_list.append(attention_probabilities)
                # Take the new ``input_feed`` vector
                # shape: (group_size, decoder_hidden_size)
                input_feed = decoder_output.squeeze(1)

            # Combine all the decoder outputs and attention probabilities
            # shape: (group_size, num_summary_tokens, decoder_hidden_size)
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            # shape: (group_size, num_summary_tokens, num_document_tokens)
            attention_probabilities = torch.cat(attention_probabilities_list, dim=1)
        else:
            # shape: (group_size, num_summary_tokens, decoder_hidden_size)
            # shape: (1, group_size, decoder_hidden_size)
            # shape: (group-size, num_summary_tokens, num_document_tokens)
            decoder_outputs, decoder_state, attention_probabilities = \
                self._decoder_forward(summary_token_embeddings, decoder_state, encoder_outputs, document_mask, token_mask)

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

        # If we are running inference, the BeamSearch interface expects the log-probabilities,
        # not the logits. Additionally, the tensor needs to be rehaped
        if is_inference:
            # shape: (group_size, summary_vocab_size)
            output_scores = torch.log_softmax(logits, dim=2).squeeze(1)
            # shape: (group_size, num_document_tokens)
            attention_probabilities = attention_probabilities.squeeze(1)
        else:
            output_scores = logits

        # Update the state dictionary
        output_state = dict(state)
        output_state['hidden'] = hidden
        output_state['memory'] = memory
        output_state['input_feed'] = input_feed
        output_state['attention'] = attention_probabilities

        return output_scores, output_state

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
        input_mask: ``torch.Tensor``, ``(batch_size, num_summary_tokens)``
            An optional mask for the summary tokens.

        Returns
        -------
        decoder_outputs: ``torch.Tensor``, ``(batch_size, num_summary_tokens, decoder_hidden_size)``
            The decoder hidden representation with attention.
        hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]``
            The final hidden states.
        attention_probabilities: ``torch.Tensor``, ``(batch_size, num_summary_tokens, num_document_tokens)``
            The attention probabilities over the document tokens for each summary token
        """
        # shape: (group_size, num_summary_tokens, decoder_hidden_size)
        # shape: (1, group_size, decoder_hidden_size)
        decoder_outputs, hidden = self.decoder(input_vectors, input_mask, hidden)

        # Add in the attention mechanism
        # shape: (group_size, num_summary_tokens, decoder_hidden_size)
        # shape: (group_size, num_summary_tokens, num_document_tokens)
        decoder_outputs, attention_probabilities = \
            self._compute_attention(encoder_outputs, encoder_mask, decoder_outputs)
        return decoder_outputs, hidden, attention_probabilities

    def _compute_attention(self,
                           encoder_outputs: torch.Tensor,
                           encoder_mask: torch.Tensor,
                           decoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention-based decoder hidden representation by first
        computing the attention scores between the encoder and decoder hidden
        states, computing the attention context via a weighted average over
        the encoder hidden states, concatenating the decoder state with the
        context, and passing the result through the attention layer to project
        it back down to the decoder hidden state size.

        Parameters
        ----------
        encoder_outputs: ``torch.Tensor``, ``(batch_size, num_document_tokens, encoder_hidden_size)``
            The output from the encoder.
        encoder_mask: ``torch.Tensor``, ``(batch_size, num_document_tokens)``
            The document token mask.
        decoder_outputs: ``torch.Tensor``, ``(batch_size, num_summary_tokens, decoder_hidden_size)``
            The output from the decoder.

        Returns
        -------
        hidden: ``torch.Tensor``, ``(batch_size, num_summary_tokens, decoder_hidden_size)``
            The new decoder hidden state representation.
        attention_probabilities: ``torch.Tensor``, ``(batch_size, num_summary_tokens, num_document_tokens)``
            The attention probabilities over the document tokens for each summary token
        """
        # Compute the attention context
        # shape: (group_size, num_summary_tokens, num_document_tokens)
        attention_scores = self.attention(decoder_outputs, encoder_outputs)
        # shape: (group_size, num_summary_tokens, num_document_tokens)
        attention_probabilities = masked_softmax(attention_scores, encoder_mask)
        # shape: (group_size, num_summary_tokens, encoder_hidden_size)
        attention_context = weighted_sum(encoder_outputs, attention_probabilities)

        # Concatenate the attention context with the decoder outputs
        # then project back to the decoder hidden size
        # shape: (group_size, num_summary_tokens, encoder_hidden_size + decoder_hidden_size)
        concat = torch.cat([attention_context, decoder_outputs], dim=2)

        # shape: (group_size, num_summary_tokens, decoder_hidden_size)
        projected_hidden = self.attention_layer(concat)
        return projected_hidden, attention_probabilities

    def _compute_loss(self,
                      logits: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss given the target tokens and logit scores.

        Parameters
        ----------
        logits: ``torch.Tensor``, ``(batch_size, num_target_tokens, vocab_size)``
            The model scores for the vocabulary items.
        targets: ``torch.Tensor``, ``(batch_size, num_target_tokens)``
            The ground-truth summary tokens.

        Returns
        -------
        loss: ``torch.Tensor``, ``(1)``
            The loss.
        cross_entropy: ``torch.Tensor``, ``(1)``
            The token-level cross-entropy.
        """
        batch_size, num_target_tokens = targets.size()
        # Reshape the inputs to the loss and compute it
        # shape: (batch_size * num_target_tokens)
        targets = targets.view(-1)
        # shape: (batch_size * num_target_tokens, summary_vocab_size)
        logits = logits.view(batch_size * num_target_tokens, -1)

        # shape: (batch_size, num_target_tokens)
        losses = self.loss(logits, targets).view(batch_size, num_target_tokens)

        # shape: (batch_size, num_target_tokens)
        targets = targets.view(batch_size, num_target_tokens)
        losses_mask = (targets != self.pad_index).float()

        loss = normalize_losses(losses, losses_mask,
                                self.instance_loss_normalization,
                                self.batch_loss_normalization)

        # Compute the token-level cross-entropy
        total_loss = (losses * losses_mask).sum()
        num_targets = losses_mask.sum()
        self.cross_entropy_metric(total_loss.item(), num_targets.item())

        return loss

    def _run_inference(self, initial_decoding_state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        summary_field_name: ``str``
            The name of the summary field in the metadata
        """
        # If we have acess to the ground-truth summaries, then we can
        # compute metrics
        batch_size = len(predictions)
        if summary_field_name in metadata[0] and self.metrics is not None:
            gold_summaries = [metadata[batch][summary_field_name] for batch in range(batch_size)]
            # shape: (batch_size, max_output_length)
            model_summaries = [prediction[0] for prediction in predictions]
            for metric in self.metrics:
                metric(gold_summaries=gold_summaries, model_summaries=model_summaries)

    @overrides
    def forward(self,
                document: Dict[str, torch.Tensor],
                metadata: List[Dict[str, Any]],
                summary: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Computes the forward pass for the ``Seq2SeqModel``.

        Parameters
        ----------
        document: ``Dict[str, torch.Tensor]``
            The document tokens.
        metadata: ``List[Dict[str, Any]]``
            The metadata for this batch.
        summary: ``Dict[str, torch.Tensor]``, optional (default = ``None``)
            The summary tokens. The dictionary must have some representation for the
            summary using the "tokens" key since it is used during training.
            If ``None``, the loss will not be calculated.
        """
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

        output_dict = {}

        # Compute the loss if we have the ground-truth summaries
        if summary is not None:
            # shape: (batch_size, num_summary_tokens - 1, summary_vocab_size)
            # shape: (batch_size, num_summary_tokens - 1, summary_vocab_size)
            logits, targets = self._run_teacher_forcing(initial_decoding_state, summary)
            output_dict['loss'] = self._compute_loss(logits, targets)

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
        batch_size = len(predictions)

        summaries = []
        for batch in range(batch_size):
            tokens = []
            prediction = predictions[batch][0]
            for i in range(len(prediction)):
                # The best prediction is in the 0th beam position
                index = prediction[i].item()
                # We skip the start, sentence start, and sentence end indices. It is ok
                # if these are ``None`` because the index should never be ``None``
                if index in [self.start_index, self.sent_start_index, self.sent_end_index]:
                    continue
                # We stop decoding if we see the end or pad symbols
                if index in [self.end_index, self.pad_index]:
                    break
                token = self.vocab.get_token_from_index(index, self.summary_namespace)
                tokens.append(token)
            summary = ' '.join(tokens)
            summaries.append(summary)

        output_dict[summary_field_name] = summaries
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get_metric(reset))
        metrics.update(self.cross_entropy_metric.get_metric(reset))
        return metrics
