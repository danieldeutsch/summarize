import torch
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, MatrixAttention, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Any, Dict, List, Optional, Tuple, Union

from summarize.modules.rnns import RNN


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
    hidden_projection_layer: ``FeedForward``, optional (default = ``None``)
        The ``hidden_projection_layer`` is applied to the final encoder hidden
        state. The output size should be equal to the decoder's hidden size. This
        is sometimes used to map a bidirectional encoder's hidden state to a single
        directional decoder's hidden state since the bidirectional encoder's hidden
        state will be twice the size as the decoder's hidden state. If ``None``,
        no projection will be used.
    memory_projection_layer: ``FeedForward``, optional (default = ``None``)
        The same as ``hidden_projection_layer`` except applied to the memory cell
        of the RNN if it has memory.
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
    beam_size: ``int``, optional (default = 1)
        The size of the beam to use for decoding.
    min_output_length: ``int``, optional (default = ``None``)
        The minimum possible summary output length, unrestricted if ``None``.
    max_output_length: ``int``, optional (default = ``None``)
        The maximum possible summary output length, unrestricted if ``None``.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 document_token_embedder: TextFieldEmbedder,
                 encoder: RNN,
                 attention: MatrixAttention,
                 attention_layer: FeedForward,
                 decoder: RNN,
                 hidden_projection_layer: Optional[FeedForward] = None,
                 memory_projection_layer: Optional[FeedForward] = None,
                 summary_token_embedder: Optional[TokenEmbedder] = None,
                 summary_namespace: str = 'tokens',
                 use_input_feeding: bool = False,
                 input_feeding_projection_layer: Optional[FeedForward] = None,
                 beam_size: int = 1,
                 min_output_length: Optional[int] = None,
                 max_output_length: Optional[int] = None,
                 metrics: Optional[List[Metric]] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self.document_token_embedder = document_token_embedder
        self.encoder = encoder
        self.attention = attention
        self.attention_layer = attention_layer
        self.decoder = decoder
        self.hidden_projection_layer = hidden_projection_layer
        self.memory_projection_layer = memory_projection_layer
        self.summary_token_embedder = summary_token_embedder
        self.summary_namespace = summary_namespace
        self.use_input_feeding = use_input_feeding
        self.input_feeding_projection_layer = input_feeding_projection_layer
        # The ``output_layer`` is applied after the attention context and decoder
        # hidden state are combined. It is used to calculate the softmax over the
        # summary vocabulary
        self.output_layer = torch.nn.Linear(decoder.get_output_dim(), vocab.get_vocab_size(summary_namespace))
        self.start_index = vocab.get_token_index(START_SYMBOL, summary_namespace)
        self.end_index = vocab.get_token_index(END_SYMBOL, summary_namespace)
        self.pad_index = vocab.get_token_index(DEFAULT_PADDING_TOKEN, summary_namespace)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_index, reduction='mean')
        self.min_output_length = min_output_length
        self.beam_search = BeamSearch(self.end_index, max_steps=max_output_length,
                                      beam_size=beam_size)
        self.metrics = metrics
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

        # Project the encoder hidden state onto the initial decoder hidden state
        if self.encoder.has_memory():
            # shape: (batch_size, encoder_hidden_size * num_directions)
            hidden, memory = hidden
        else:
            # We still create the memory even if the decoder is a GRU because the
            # beam search code will try to reshape every tensor and it will fail if
            # it is `None`. However, it won't be used in computation. If this ever
            # gets changed to be initialized to the encoder's memory, we should also
            # add its own projection layer.
            # shape: (batch_size, encoder_hidden_size * num_directions)
            memory = hidden.new_zeros(hidden.size())

        if self.hidden_projection_layer is not None:
            # shape: (batch_size, decoder_hidden_size)
            hidden = self.hidden_projection_layer(hidden)
        if self.memory_projection_layer is not None:
            # shape: (batch_size, decoder_hidden_size)
            memory = self.hidden_projection_layer(memory)

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
                      state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        if summary_tokens.dim() == 1:
            # shape: (group_size, 1)
            summary_tokens = summary_tokens.unsqueeze(-1)
        group_size, num_summary_tokens = summary_tokens.size()

        # Unpack everything from the input state
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
                decoder_output, decoder_state = \
                    self._decoder_forward(input_embedding, decoder_state, encoder_outputs, document_mask)

                # Save the decoder output
                decoder_outputs.append(decoder_output)
                # Take the new ``input_feed`` vector
                # shape: (group_size, decoder_hidden_size)
                input_feed = decoder_output.squeeze(1)

            # Combine all the decoder outputs
            # shape: (group_size, num_summary_tokens, decoder_hidden_size)
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
        else:
            # shape: (group_size, num_summary_tokens, decoder_hidden_size)
            # shape: (1, group_size, decoder_hidden_size)
            decoder_outputs, decoder_state = \
                self._decoder_forward(summary_token_embeddings, decoder_state, encoder_outputs, document_mask)

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

        # Reshape the logits if there was only 1 summary token. This is typically
        # because it's being called from `BeamSearch`
        if num_summary_tokens == 1:
            logits = logits.squeeze(1)

        # Update the state dictionary
        output_state = dict(state)
        output_state['hidden'] = hidden
        output_state['memory'] = memory
        output_state['input_feed'] = input_feed

        return logits, output_state

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
                         encoder_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        Returns
        -------
        decoder_outputs: ``torch.Tensor``, ``(batch_size, num_summary_tokens, decoder_hidden_size)``
            The decoder hidden representation with attention.
        hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]``
            The final hidden states.
        """
        # Pass the tokens through the decoder. Masking is not necessary because
        # if this method is used for beam search, anything after <eos> will be
        # discarded. If it's used for computing the loss, we will ignore anything
        # after <eos> when computing the loss function. In neither case do we
        # care about having an incorrect hidden state.
        #
        # shape: (group_size, num_summary_tokens, decoder_hidden_size)
        # shape: (1, group_size, decoder_hidden_size)
        decoder_outputs, hidden = self.decoder(input_vectors, None, hidden)

        # Add in the attention mechanism
        # shape: (group_size, num_summary_tokens, decoder_hidden_size)
        decoder_outputs = self._compute_attention(encoder_outputs, encoder_mask, decoder_outputs)
        return decoder_outputs, hidden

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
        return projected_hidden

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
        """
        batch_size, num_target_tokens = targets.size()
        # Reshape the inputs to the loss and compute it
        # shape: (batch_size * num_target_tokens)
        targets = targets.view(-1)
        # shape: (batch_size * num_target_tokens, summary_vocab_size)
        logits = logits.view(batch_size * num_target_tokens, -1)
        loss = self.loss(logits, targets)
        return loss

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
            self.beam_search.search(initial_predictions, initial_decoding_state,
                                    self._decoder_step)
        return predictions, log_probabilities

    def _update_metrics(self,
                        predictions: torch.Tensor,
                        metadata: List[Dict[str, Any]]) -> None:
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
        batch_size = predictions.size(0)
        if 'summary' in metadata[0] and self.metrics is not None:
            gold_summaries = [metadata[batch]['summary'] for batch in range(batch_size)]
            # shape: (batch_size, max_output_length)
            model_summaries = predictions[:, 0, :]
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
        if not self.training:
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
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        predictions = output_dict.pop('predictions')
        batch_size, beam_size, max_output_length = predictions.size()

        summaries = []
        for batch in range(batch_size):
            tokens = []
            for i in range(max_output_length):
                # The best prediction is in the 0th beam position
                index = predictions[batch, 0, i].item()
                if index == self.start_index or self.end_index:
                    # The end index also separates sentences, so it does not
                    # mean we should stop.
                    continue
                if index == self.pad_index:
                    break
                token = self.vocab.get_token_from_index(index, self.summary_namespace)
                tokens.append(token)
            summary = ' '.join(tokens)
            summaries.append(summary)

        output_dict['summary'] = summaries
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get_metric(reset))
        return metrics
