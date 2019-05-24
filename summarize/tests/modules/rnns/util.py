import random
import torch
from allennlp.modules import Seq2SeqEncoder

from summarize.modules.rnns import GRU, LSTM


def get_random_inputs(batch_size: int, sequence_length: int, input_size: int):
    """
    Creates and returns random masked input data for an RNN.
    """
    input_data = torch.randn(batch_size, sequence_length, input_size)
    mask = torch.ones(batch_size, sequence_length, dtype=torch.uint8)
    # Start with 1 to make sure one of the inputs is not masked at all
    for i in range(1, batch_size):
        index = random.randint(1, sequence_length)
        mask[i, index:] = 0
    return input_data, mask


def get_rnns(rnn_type: str, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool):
    """
    Creates and returns an equivalent AllenNLP ``Seq2SeqEncoder`` and ``RNN`` RNNs.
    """
    assert num_layers in [1, 2]
    assert rnn_type in ['gru', 'lstm']
    seq2seq_encoder = Seq2SeqEncoder.by_name(rnn_type)(input_size=input_size, hidden_size=hidden_size,
                                                       num_layers=num_layers, bidirectional=bidirectional)
    if rnn_type == 'gru':
        rnn = GRU(input_size, hidden_size, num_layers, bidirectional)
    else:
        rnn = LSTM(input_size, hidden_size, num_layers, bidirectional)

    rnn.rnn.weight_ih_l0[:] = seq2seq_encoder._module.weight_ih_l0[:]
    rnn.rnn.weight_hh_l0[:] = seq2seq_encoder._module.weight_hh_l0[:]
    rnn.rnn.bias_ih_l0[:] = seq2seq_encoder._module.bias_ih_l0[:]
    rnn.rnn.bias_hh_l0[:] = seq2seq_encoder._module.bias_hh_l0[:]
    if bidirectional:
        rnn.rnn.weight_ih_l0_reverse[:] = seq2seq_encoder._module.weight_ih_l0_reverse[:]
        rnn.rnn.weight_hh_l0_reverse[:] = seq2seq_encoder._module.weight_hh_l0_reverse[:]
        rnn.rnn.bias_ih_l0_reverse[:] = seq2seq_encoder._module.bias_ih_l0_reverse[:]
        rnn.rnn.bias_hh_l0_reverse[:] = seq2seq_encoder._module.bias_hh_l0_reverse[:]

    if num_layers == 2:
        rnn.rnn.weight_ih_l1[:] = seq2seq_encoder._module.weight_ih_l1[:]
        rnn.rnn.weight_hh_l1[:] = seq2seq_encoder._module.weight_hh_l1[:]
        rnn.rnn.bias_ih_l1[:] = seq2seq_encoder._module.bias_ih_l1[:]
        rnn.rnn.bias_hh_l1[:] = seq2seq_encoder._module.bias_hh_l1[:]
        if bidirectional:
            rnn.rnn.weight_ih_l1_reverse[:] = seq2seq_encoder._module.weight_ih_l1_reverse[:]
            rnn.rnn.weight_hh_l1_reverse[:] = seq2seq_encoder._module.weight_hh_l1_reverse[:]
            rnn.rnn.bias_ih_l1_reverse[:] = seq2seq_encoder._module.bias_ih_l1_reverse[:]
            rnn.rnn.bias_hh_l1_reverse[:] = seq2seq_encoder._module.bias_hh_l1_reverse[:]

    return seq2seq_encoder, rnn
