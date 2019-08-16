import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Any, Dict, List, Optional

from summarize.modules.sentence_extractors import SentenceExtractor


@Model.register('sds-extractive-baseline')
class ExtractiveBaselineModel(Model):
    """
    The ``ExtractiveBaselineModel`` is a standard extractive model for single
    document summarization. The document sentences are first encoded with a
    ``Seq2VecEncoder``. Then, a ``SentenceExtractor`` computes the probability
    that each individual sentence should be extracted. Inference is done by
    sorting the sentences by their corresponding extraction probabilities and then
    greedily taking sentences until a maximum word budget has been reached.

    The model is based on Kedzie et al. (2018) (https://arxiv.org/pdf/1810.12343.pdf).

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
                 max_words: int,
                 dropout: float = 0.0,
                 metrics: Optional[List[Metric]] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.token_embedder = token_embedder
        self.sentence_encoder = sentence_encoder
        self.sentence_extractor = sentence_extractor
        self.max_words = max_words
        self.dropout = torch.nn.Dropout(dropout)
        self.metrics = metrics
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        initializer(self)

    def _get_sentence_extraction_scores(self, document: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the sentence extraction scores from the input document.

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

        # Pass the encodings through the sentence extractor
        # shape: (batch_size, num_sents)
        extraction_scores = self.sentence_extractor(sentence_encodings, sentence_mask)
        return extraction_scores, sentence_mask, sentence_lengths

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
                    if cost >= self.max_words:
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
            The metadata for each instance. If "summary" is included in the metadata,
            the Rouge score can be calculated.
        """
        # If we have access to the ground-truth summary, we can compute metrics
        if 'summary' in metadata[0] and self.metrics is not None:
            batch_size = sentence_mask.size(0)

            # Extract the gold and model summaries
            gold_summaries, model_summaries = [], []
            for batch in range(batch_size):
                gold_summary = metadata[batch]['summary']
                gold_summaries.append(gold_summary)

                input_document = metadata[batch]['document']
                model_summary = [input_document[index] for index in predicted_indices[batch]]
                model_summaries.append(model_summary)

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
                metric(gold_summaries=gold_summaries,
                       model_summaries=model_summaries,
                       gold_labels=gold_labels,
                       model_labels=model_labels,
                       mask=flat_mask)

    @overrides
    def forward(self,
                document: Dict[str, torch.Tensor],
                metadata: List[Dict[str, Any]],
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, num_sents)
        # shape: (batch_size, num_sents)
        # shape: (batch_size, num_sents)
        extraction_scores, sentence_mask, sentence_lengths = self._get_sentence_extraction_scores(document)

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
