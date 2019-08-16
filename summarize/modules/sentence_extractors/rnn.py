import torch
from allennlp.common.checks import ConfigurationError
from allennlp.modules import FeedForward, Seq2SeqEncoder
from overrides import overrides

from summarize.modules.sentence_extractors import SentenceExtractor


@SentenceExtractor.register('rnn')
class RNNSentenceExtractor(SentenceExtractor):
    """
    The RNNSentenceExtractor calculates extraction scores by running an RNN
    over the sentence representations followed by a feed-forward layer
    on the new hidden states.

    Parameters
    ----------
    rnn:
        The RNN to use (or any Seq2SeqEncoder)
    feed_forward:
        The feed-forward layer, which must have output dimension 1.
    dropout:
        The dropout to apply on the RNN hidden states.
    """
    def __init__(self,
                 rnn: Seq2SeqEncoder,
                 feed_forward: FeedForward,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.rnn = rnn
        self.feed_forward = feed_forward
        self.dropout = torch.nn.Dropout(dropout)

        if rnn.get_output_dim() != feed_forward.get_input_dim():
            raise ConfigurationError('The RNN and feed-forward layers have incompatible dimensions')
        if feed_forward.get_output_dim() != 1:
            raise ConfigurationError('The feed-foward network must have output size 1')

    @overrides
    def forward(self,
                sentence_encodings: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, num_sents, hidden_size)
        hidden_encodings = self.rnn(sentence_encodings, mask)
        hidden_encodings = self.dropout(hidden_encodings)
        # shape: (batch_size, num_sents)
        extraction_scores = self.feed_forward(hidden_encodings).squeeze(-1)
        return extraction_scores
