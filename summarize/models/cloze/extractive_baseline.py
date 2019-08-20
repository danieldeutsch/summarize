import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, MatrixAttention, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Any, Dict, List, Optional, Tuple

from summarize.modules.sentence_extractors import SentenceExtractor


@Model.register('cloze-extractive-baseline')
class ClozeExtractiveBaselineModel(Model):
    """
    The ``ClozeExtractiveBaselineModel`` is a baseline extractive model for the
    single-document cloze task.

    Parameters
    ----------
    token_embedder:
        The embedder for the document tokens
    sentence_encoder:
        The encoder which creates a sentence representation based on the token embeddings
    sentence_extractor:
        The module which computes the sentence extraction probabilities
    max_words:
        The maximum number of words that the extractive model is allowed to select
        during inference.
    dropout:
        The dropout probability
    metrics:
        The metrics which should be computed during validation.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 sentence_extractor: SentenceExtractor,
                 topic_encoder: Seq2VecEncoder = None,
                 topic_layer: FeedForward = None,
                 context_encoder: Seq2SeqEncoder = None,
                 attention: MatrixAttention = None,
                 attention_layer: FeedForward = None,
                 use_topics: bool = False,
                 use_context: bool = False,
                 dropout: float = 0.0,
                 max_words: int = None,
                 max_sents: int = None,
                 metrics: Optional[List[Metric]] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.token_embedder = token_embedder
        self.sentence_encoder = sentence_encoder
        self.sentence_extractor = sentence_extractor
        self.topic_encoder = topic_encoder
        self.topic_layer = topic_layer
        self.context_encoder = context_encoder
        self.attention = attention
        self.attention_layer = attention_layer
        self.use_topics = use_topics
        self.use_context = use_context
        self.dropout = torch.nn.Dropout(dropout)
        self.max_words = max_words
        self.max_sents = max_sents
        self.metrics = metrics
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        initializer(self)

        if self.max_words is None and self.max_sents is None:
            raise Exception('`max_words` or `max_sents` must be provided')

    def _encode_document(self, document: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Computes the representation for each sentence in the input document.

        Returns
        -------
        extraction_scores: (batch_size, num_sents)
            The raw extraction scores (not probabilities)
        sentence_mask: (batch_size, num_sents)
            The binary sentence mask
        sentence_lengths: (batch_size, num_sents)
            The length of each sentence in tokens
        """
        batch_size, num_sents, num_tokens = document['tokens'].size()

        # Compute the token- and sentence-level masks.
        # shape: (batch_size, num_sents, num_tokens)
        token_mask = get_text_field_mask(document, num_wrapping_dims=1)
        # shape: (batch_size, num_sents)
        sentence_lengths = token_mask.sum(dim=2)
        # shape: (batch_size, num_sents)
        sentence_mask = (sentence_lengths > 0).long()

        # Get the token embeddings
        # shape: (batch_size, num_sents, num_tokens, embed_size)
        token_embeddings = self.token_embedder(document)
        token_embeddings = self.dropout(token_embeddings)

        # Reshape the token embeddings and mask to be passed through the sentence encoder
        # shape: (batch_size * num_sents, num_tokens)
        token_mask = token_mask.view(batch_size * num_sents, num_tokens)
        # shape: (batch_size * num_sents, num_tokens, embed_size)
        token_embeddings = token_embeddings.view(batch_size * num_sents, num_tokens, -1)

        # Get the sentence encodings
        # shape: (batch_size * num_sents, hidden_size)
        sentence_encodings = self.sentence_encoder(token_embeddings, token_mask)
        sentence_encodings = self.dropout(sentence_encodings)

        # Reshape to setup the sentence extractor
        # shape: (batch_size, num_sents, hidden_size)
        sentence_encodings = sentence_encodings.view(batch_size, num_sents, -1)
        return sentence_encodings, sentence_mask, sentence_lengths

    def _encode_topics(self, topics: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Returns
        -------
        topic_encodings: (batch_size, num_topics, hidden_dim)
            The encoding for each topic
        topic_mask: (batch_size, num_topics)
            The topic mask
        """
        if self.topic_encoder is None:
            raise Exception('`topic_encoder` cannot be `None` to encode the topics')

        batch_size, num_topics, num_tokens = topics['tokens'].size()
        # shape: (batch_size, num_topics, num_tokens, embed_size)
        embeddings = self.token_embedder(topics)
        embeddings = self.dropout(embeddings)
        # shape: (batch_size, num_topics, num_tokens)
        mask = get_text_field_mask(topics, num_wrapping_dims=1)

        # Reshape the embeddings and mask to run through the topic encoder
        # shape: (batch_size * num_topics, num_tokens, embed_size)
        embeddings = embeddings.view(batch_size * num_topics, num_tokens, -1)
        # shape: (batch_size * num_topics, num_tokens)
        token_mask = mask.view(batch_size * num_topics, num_tokens)

        # shape: (batch_size * num_topics, hidden_size)
        encodings = self.topic_encoder(embeddings, token_mask)
        encodings = self.dropout(encodings)
        # Reshape the encodings so there is a list of topic encodings per batch
        # shape: (batch_size, num_topics, hidden_size)
        encodings = encodings.view(batch_size, num_topics, -1)
        # Create a topic_mask
        # shape: (batch_size, num_topics)
        topic_mask = (token_mask.long().sum(dim=1) > 0).long().view(batch_size, num_topics)

        # Potentially pass the topics through a feed forward layer in order to
        # be the same size as the context encodings
        if self.topic_layer is not None:
            # shape: (batch_size, num_topics, hidden_size)
            encodings = self.topic_layer(encodings)

        return encodings, topic_mask

    def _encode_context(self, context: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Returns
        -------
        context_encodings: (batch_size, num_context_tokens, hidden_dim)
            The encoding for each context token
        mask: (batch_size, num_context_tokens)
            The token-level mask
        """
        if self.context_encoder is None:
            raise Exception('`context_encoder` cannot be `None` to encode the topics')

        # shape: (batch_size, num_context_tokens)
        mask = get_text_field_mask(context)
        # shape: (batch_size, num_context_tokens, embed_size)
        embeddings = self.token_embedder(context)
        embeddings = self.dropout(embeddings)
        # shape: (batch_size, num_context_tokens, hidden_size)
        encodings = self.context_encoder(embeddings, mask)
        encodings = self.dropout(encodings)
        return encodings, mask

    def _compute_attention(self,
                           sentence_encodings: torch.Tensor,
                           context_encodings: torch.Tensor,
                           context_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes new sentence encodings using an attention mechanism between
        the original sentence encodings and some context encodings. The context
        encodings are not necessarily the context in the cloze task sense, but
        any vector over which the attention should be computed.

        Parameters
        ----------
        sentence_encodings: (batch_size, num_sents, hidden_dim)
            The original sentence encodings
        context_encodings: (batch_size, num_contexts, hidden_dim)
            The representation of each context item
        context_mask: (batch_size, num_contexts)
            The context item mask

        Returns
        -------
        The new sentence encodings: (batch_size, num_sents, hidden_dim)
        """
        if self.attention is None or self.attention_layer is None:
            raise Exception('`attention` and `attention_layer` must not be `None` to use attention')

        # shape: (batch_size, num_sents, num_context_tokens)
        attention_scores = self.attention(sentence_encodings, context_encodings)
        # shape: (batch_size, num_sents, num_context_tokens)
        attention_probabilities = masked_softmax(attention_scores, context_mask)
        # shape: (batch_size, num_sents, hidden_size)
        attention_context = weighted_sum(context_encodings, attention_probabilities)

        # Concatenate the attention context with the sentence encodings
        # then project back to the sentence encoder hidden size
        # shape: (batch_size, num_sents, hidden_size * 2)
        concat = torch.cat([attention_context, sentence_encodings], dim=2)
        # shape: (batch_size, num_sents, hidden_size)
        projected_hidden = self.attention_layer(concat)
        return projected_hidden

    def _compute_loss(self,
                      extraction_scores: torch.Tensor,
                      sentence_mask: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the binary cross entropy loss using the extraction scores
        and ground-truth labels.

        Parameters
        ----------
        extraction_scores: (batch_size, num_sents)
        sentence_mask: (batch_size, num_sents)
        labels: (batch_size, num_sents)

        Returns
        -------
        The loss.
        """
        # Calculate the weights for each label. The negative labels recieve weight 1,
        # and the positive labels recieve weight num_negative_labels / num_positive_labels.
        # shape: (batch_size)
        num_positive_labels = labels.sum(dim=1)
        # shape: (batch_size)
        num_negative_labels = sentence_mask.sum(dim=1).float() - num_positive_labels
        # shape: (batch_size)
        positive_weights = num_negative_labels / num_positive_labels
        # shape: (batch_size)
        negative_weights = positive_weights.new_ones(extraction_scores.size(0), dtype=torch.float32)
        # shape: (batch_size, 2)
        label_weights = torch.stack([negative_weights, positive_weights], dim=1)

        # Get the weight for each sentence by indexing into the corresponding
        # label weights followed by an element-wise masking
        # shape: (batch_size, num_sents)
        sentence_weights = torch.gather(label_weights, 1, labels.long())
        # shape: (batch_size, num_sents)
        sentence_weights = sentence_weights * sentence_mask.float()

        # Compute the loss. Since the label weights are specific to each
        # instance, we have to manually do the loss reduction ourselves.
        # shape: (batch_size, num_sents)
        loss = self.loss(extraction_scores, labels)
        # shape: (batch_size, num_sents)
        loss = loss * sentence_weights
        # shape: (1)
        num_sentences = sentence_mask.float().sum()
        # shape: (1)
        loss = loss.sum() / num_sentences
        return loss

    def _predict(self,
                 extraction_scores: torch.Tensor,
                 sentence_mask: torch.Tensor,
                 sentence_lengths: torch.Tensor) -> List[List[int]]:
        """
        Runs inference to select the indices which correspond to the predicted
        summary.

        Parameters
        ----------
        extraction_scores: (batch_size, num_sents)
        sentence_mask: (batch_size, num_sents)
        sentence_lengths: (batch_size, num_sents)

        Returns
        -------
        The selected indices, represented as a ``List[List[int]]``.
        """
        batch_size = extraction_scores.size(0)
        argsort = torch.argsort(extraction_scores, dim=1, descending=True)
        batched_prediction = []
        for batch in range(batch_size):
            cost = 0
            prediction = []
            for index in argsort[batch]:
                index = index.item()
                if sentence_mask[batch, index] == 1:
                    prediction.append(index)
                    cost += sentence_lengths[batch, index].item()
                    # We can't take part of the sentence easily, so we keep
                    # taking sentences until we hit the maximum number of words.
                    # The metric computation should take care of not using the
                    # full sentence if we take more words than we are allowed.
                    if self.max_words is not None and cost >= self.max_words:
                        break
                    if self.max_sents is not None and len(prediction) >= self.max_sents:
                        break
            # Put the sentences in order that they appear in the document
            prediction.sort()
            batched_prediction.append(prediction)
        return batched_prediction

    def _update_metrics(self,
                        predicted_indices: List[List[int]],
                        sentence_mask: torch.Tensor,
                        labels: torch.Tensor,
                        metadata: List[Dict[str, Any]]) -> None:
        """
        Updates the running metrics based on the predictions.

        Parameters
        ----------
        predicted_indices:
            The output from inference, indexed by the batch and then the sentence number
        sentence_mask: (batch_size, num_sents)
        labels: (batch_size, num_sents)
        metadata:
            The metadata for each instance. If "cloze" is included in the metadata,
            the Rouge score can be calculated.
        """
        # If we have access to the ground-truth cloze, we can compute metrics
        if 'cloze' in metadata[0] and self.metrics is not None:
            batch_size = sentence_mask.size(0)

            # Extract the gold and model summaries
            gold_clozes, model_clozes = [], []
            for batch in range(batch_size):
                gold_cloze = metadata[batch]['cloze']
                gold_clozes.append(gold_cloze)

                input_document = metadata[batch]['document']
                model_cloze = [input_document[index] for index in predicted_indices[batch]]
                model_clozes.append(model_cloze)

            # Create a label matrix for the model predictions and put the
            # labels into the matrix
            # shape: (batch_size, num_sents)
            model_labels = labels.new_zeros(labels.size(), dtype=torch.long)
            for batch in range(batch_size):
                for index in predicted_indices[batch]:
                    model_labels[batch, index] = 1

            # Flatten the labels for the metrics
            gold_labels = labels.view(-1)
            model_labels = model_labels.view(-1)
            flat_mask = sentence_mask.view(-1)

            for metric in self.metrics:
                metric(gold_summaries=gold_clozes,
                       model_summaries=model_clozes,
                       gold_labels=gold_labels,
                       model_labels=model_labels,
                       mask=flat_mask)

    @overrides
    def forward(self,
                document: Dict[str, torch.Tensor],
                topics: Dict[str, torch.Tensor],
                context: Dict[str, torch.Tensor],
                metadata: List[Dict[str, Any]],
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, num_sents)
        # shape: (batch_size, num_sents)
        # shape: (batch_size, num_sents)
        sentence_encodings, sentence_mask, sentence_lengths = self._encode_document(document)

        if self.use_topics:
            # shape: (batch_size, num_topics, hidden_size)
            # shape: (batch_size, num_topics)
            topic_encodings, topic_mask = self._encode_topics(topics)

        if self.use_context:
            # shape: (batch_size, num_context_tokens, hidden_size)
            # shape: (batch_size, num_context_tokens)
            context_encodings, context_mask = self._encode_context(context)

        # Attention is only computed if we use the topic or context
        if self.use_topics or self.use_context:
            # The input to attention depends on if we use the topic, context, or both
            if self.use_topics and self.use_context:
                # shape: (batch_size, num_topics + num_context_tokens, hidden_size)
                attention_input_encodings = torch.cat([topic_encodings, context_encodings], dim=1)
                # shape: (batch_size, num_topics + num_context_tokens)
                attention_input_mask = torch.cat([topic_mask, context_mask], dim=1)
            elif self.use_topics:
                attention_input_encodings = topic_encodings
                attention_input_mask = topic_mask
            elif self.use_context:
                attention_input_encodings = context_encodings
                attention_input_mask = context_mask
            else:
                raise Exception(f'Illegal state')

            # Compute new sentence representations with attention
            # shape: (batch_size, num_sents, hidden_size)
            sentence_encodings = \
                self._compute_attention(sentence_encodings, attention_input_encodings, attention_input_mask)

        # shape: (batch_size, num_sents)
        extraction_scores = self.sentence_extractor(sentence_encodings, sentence_mask)

        output_dict = {}
        if labels is not None:
            output_dict['loss'] = self._compute_loss(extraction_scores, sentence_mask, labels)

        if not self.training:
            predicted_indices = self._predict(extraction_scores, sentence_mask,
                                              sentence_lengths)
            output_dict['predicted_indices'] = predicted_indices
            self._update_metrics(predicted_indices, sentence_mask, labels, metadata)

        output_dict['metadata'] = metadata
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get_metric(reset))
        return metrics
