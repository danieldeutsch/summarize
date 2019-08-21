"""
Utilities for RNN computation, including created packed sequences that are used
for proper masking in RNNs.
"""
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Union


def pack_sequence(X: torch.Tensor,
                  mask: torch.Tensor,
                  hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None) -> PackedSequence:
    """
    Prepares a tensor for input to an RNN.

    Parameters
    ----------
    X: ``torch.Tensor``, ``(batch_size, sequence_length, input_size)``
        The input data.
    mask: ``torch.Tensor``, ``(batch_size, sequence_length)``
        The input mask.

    Returns
    -------
    The packed sequence that can be passed to an RNN.
    """
    assert X.dim() == 3
    assert X.size(0) == mask.size(0)
    assert X.size(1) == mask.size(1)

    # Sort X from longest to smallest lengths
    lengths = mask.float().sum(dim=-1)
    seq_lengths, seq_idx = lengths.sort(0, descending=True)
    seq_lengths = seq_lengths.int().data.tolist()
    X = X[seq_idx]

    if hidden is not None:
        if isinstance(hidden, torch.Tensor):
            hidden = hidden[:, seq_idx, :]
        else:
            h, c = hidden
            h = h[:, seq_idx, :]
            c = c[:, seq_idx, :]
            hidden = (h, c)

    packed = pack_padded_sequence(X, seq_lengths, batch_first=True)
    return packed, hidden


def unpack_sequence(packed_sequence: PackedSequence,
                    hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpacks the output of the RNN into normal tensors.

    Parameters
    ----------
    packed_sequence: ``PackedSequence``
        The output from the RNN.
    hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]``, ``(num_layers * num_directions, batch_size, hidden_size)``
        The output hidden state of the RNN.
    mask: ``torch.Tensor``, ``(batch_size, sequence_length)``
        The input data mask.

    Returns
    -------
    scores: ``torch.Tensor``, ``(batch_size, sequence_length, hidden_size)``
        The outputs from the RNN
    hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]``, ``(num_layers * num_directions, batch_size, hidden_size)``
        The hidden states.
    """
    unpacked, _ = pad_packed_sequence(packed_sequence, batch_first=True)

    lengths = mask.float().sum(dim=-1)
    _, seq_idx = lengths.sort(0, descending=True)
    _, original_idx = seq_idx.sort(0, descending=False)

    scores = unpacked[original_idx]

    # If the RNN is a GRU, the hidden will just be
    # a tensor instead of a tuple
    if isinstance(hidden, torch.Tensor):
        hidden = hidden[:, original_idx, :]
    else:
        h, c = hidden
        h = h[:, original_idx, :]
        c = c[:, original_idx, :]
        hidden = (h, c)
    return scores, hidden
