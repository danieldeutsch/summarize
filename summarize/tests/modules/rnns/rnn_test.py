import torch
import unittest

from summarize.tests.modules.rnns import util


class TestRNN(unittest.TestCase):
    def test_rnn_seq2seq_encoder_are_identical(self):
        batch_size = 3
        sequence_length = 11
        input_size = 5
        hidden_size = 7
        num_layers = 2
        bidirectional = True

        input_data, mask = util.get_random_inputs(batch_size, sequence_length, input_size)
        seq2seq_encoder, rnn = util.get_rnns('gru', input_size, hidden_size, num_layers, bidirectional)

        # First, compare without any masking
        expected_outputs = seq2seq_encoder(input_data, None)
        actual_outputs, _ = rnn(input_data, None)
        assert torch.equal(expected_outputs, actual_outputs)

        # Now with the masking
        expected_outputs = seq2seq_encoder(input_data, mask)
        actual_outputs, _ = rnn(input_data, mask)
        assert torch.equal(expected_outputs, actual_outputs)

    def test_rnn_seq2seq_encoder_are_identical_for_loop(self):
        # Tests the Seq2SeqEncoder versus the RNN to make sure that when the
        # RNN is applied with a for loop that the final outputs are the same
        batch_size = 3
        sequence_length = 11
        input_size = 5
        hidden_size = 7
        num_layers = 2
        bidirectional = False

        input_data, mask = util.get_random_inputs(batch_size, sequence_length, input_size)
        seq2seq_encoder, rnn = util.get_rnns('gru', input_size, hidden_size, num_layers, bidirectional)

        expected_outputs = seq2seq_encoder(input_data, None)
        actual_outputs = []
        hidden = None
        for i in range(sequence_length):
            input_step = input_data[:, i, :].unsqueeze(1)
            actual_output, hidden = rnn(input_step, None, hidden)
            actual_outputs.append(actual_output)
        actual_outputs = torch.cat(actual_outputs, dim=1)
        assert torch.equal(expected_outputs, actual_outputs)
