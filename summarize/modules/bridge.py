import torch
from allennlp.common import FromParams
from allennlp.modules import FeedForward
from typing import List, Tuple, Union


class Bridge(torch.nn.Module, FromParams):
    """
    The ``Bridge`` module is used to pass the final encoder states of an RNN
    through a linear layer before initializing the decoder RNN. There is a separate
    linear layer for each hidden state (e.g., 2 for LSTM, 1 for GRU).

    This implementation allows for using the same parameters for both the
    forward and backward RNNs, as in the OpenNMT implementation of the bridge.
    The input to forward is expected to be tensors of size ``(batch_size, encoder_hidden_size)``.
    If the encoder is bidirectional, both directions' hidden states are expected to be
    already concatenated together. If ``share_bidirectional_parameters`` is set, the input
    tensors are reshaped into ``(batch_size, 2, encoder_hidden_size // 2)``-sized tensors,
    passed through the linear layers, and then reshaped back to the expected output sizes.

    Parameters
    ----------
    layers: ``List[FeedForward]``
        The linear layers, one for each hidden state of the RNN.
    share_bidirectional_parameters: ``bool``, optional (default = ``False``)
        Indicates whether the forward and backward directions should share the
        same linear layer parameters. If a single-directional encoder is used,
        this should be ``False``.
    """
    def __init__(self,
                 layers: List[FeedForward],
                 share_bidirectional_parameters: bool = False) -> None:
        super().__init__()
        # The layers have to be mapped as a ``ModuleList`` or else they won't
        # be detected as parameters for the model
        self.layers = torch.nn.ModuleList(layers)
        self.share_bidirectional_parameters = share_bidirectional_parameters

    def forward(self, hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Applies the bridge layer to the final encoder hidden states. The input
        to the ``forward`` is expected to have already been reshaped for
        initializing the decoder.

        Parameters
        ----------
        hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, ...]]``
            A tensor or tuple of tensors, each of size ``(batch_size, encoder_hidden_size)``

        Returns
        -------
        A tensor or tuple of tensors passed through the bridge layer.
        """
        # First, map the non-tuple version to a 1-tuple for easier processing.
        # We will undo this at the end
        if not isinstance(hidden, tuple):
            hidden = (hidden,)

        batch_size, hidden_size = hidden[0].size()

        # If we are going to share parameters across the forward and backward
        # directions, then we need to separate them in the tensors
        if self.share_bidirectional_parameters:
            # shape: (batch_size, 2, encoder_hidden_size // 2)
            hidden = tuple(h.view(batch_size, 2, -1) for h in hidden)

        # Apply the bridge
        output = tuple(layer(h) for layer, h in zip(self.layers, hidden))

        # Reshape the tensors if the parameters are shared
        if self.share_bidirectional_parameters:
            # shape: (batch_size, decoder_hidden_size)
            output = tuple(h.view(batch_size, -1) for h in output)

        # Undo the tuple if there's only 1 element
        if len(output) == 1:
            output = output[0]
        return output
