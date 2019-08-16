import torch
import unittest
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn import Activation
from torch.nn import GRU

from summarize.modules.sentence_extractors import RNNSentenceExtractor


class RNNSentenceExtractorTest(unittest.TestCase):
    def test_rnn_sentence_extractor(self):
        # Hyperparameters
        batch_size = 3
        num_sents = 5
        input_hidden_size = 7
        hidden_size = 11

        # Setup a model
        gru = GRU(input_size=input_hidden_size,
                  hidden_size=hidden_size,
                  bidirectional=True,
                  batch_first=True)
        rnn = PytorchSeq2SeqWrapper(gru)
        feed_forward = FeedForward(input_dim=hidden_size * 2,
                                   num_layers=2,
                                   hidden_dims=[10, 1],
                                   activations=[Activation.by_name('tanh')(), Activation.by_name('linear')()])
        extractor = RNNSentenceExtractor(rnn, feed_forward)

        # Setup some dummy data
        sentence_encodings = torch.randn(batch_size, num_sents, input_hidden_size)
        mask = torch.ones(batch_size, num_sents)

        # Pass the data through and verify the size of the output
        extraction_scores = extractor(sentence_encodings, mask)
        assert extraction_scores.size() == (batch_size, num_sents)
