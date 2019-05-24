import torch
import unittest
from allennlp.nn.util import get_final_encoder_states

from summarize.tests.modules.rnns import util


class TestLSTM(unittest.TestCase):
    def test_lstm_remap_hidden(self):
        batch_size = 3
        sequence_length = 11
        input_size = 5
        hidden_size = 7
        num_layers = 1
        bidirectional = True

        input_data, mask = util.get_random_inputs(batch_size, sequence_length, input_size)
        seq2seq_encoder, rnn = util.get_rnns('lstm', input_size, hidden_size, num_layers, bidirectional)

        # Ensure the final encoder states are the same, with and without masking
        ones_mask = torch.ones(mask.size())
        encoder_outputs = seq2seq_encoder(input_data, None)
        expected_hidden = get_final_encoder_states(encoder_outputs, ones_mask, bidirectional)
        _, hidden = rnn(input_data, None)
        actual_hidden = rnn.reshape_hidden_for_decoder(hidden)
        actual_hidden, _ = actual_hidden
        assert (torch.abs(expected_hidden - actual_hidden) < 1e-5).all()

        encoder_outputs = seq2seq_encoder(input_data, mask)
        expected_hidden = get_final_encoder_states(encoder_outputs, mask, bidirectional)
        _, hidden = rnn(input_data, mask)
        actual_hidden = rnn.reshape_hidden_for_decoder(hidden)
        actual_hidden, _ = actual_hidden
        assert (torch.abs(expected_hidden - actual_hidden) < 1e-5).all()
