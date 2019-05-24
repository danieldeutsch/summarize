import torch
from overrides import overrides
from typing import Tuple

from summarize.modules.rnns import RNN


@RNN.register('lstm')
class LSTM(RNN):
    """
    A wrapper around the ``torch.nn.LSTM`` module.

    Parameters
    ----------
    input_size: ``int``, required
        The size of the input dimension.
    hidden_size: ``hidden_size``, required
        The size of the hidden dimension. If bidirectional, each direction will
        be this hidden size.
    num_layers: ``num_layers``, required
        The number of layers.
    bidirectional: ``bool``, required
        Indicates if the RNN is bidirectional or not.
    dropout: ``float``, optional (default = ``0.0``)
        The dropout parameter in between RNN layers.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0.0) -> None:
        rnn = torch.nn.LSTM(input_size, hidden_size,
                            bidirectional=bidirectional,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout)
        super().__init__(input_size, hidden_size, num_layers, bidirectional, rnn)

    @overrides
    def has_memory(self) -> bool:
        return True

    @overrides
    def reshape_hidden_for_decoder(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_layers != 1:
            # Not entirely sure what to do here. AllenNLP just returns the last
            # layer, but I don't know if that's correct.
            raise NotImplementedError

        # Split the hidden from the memory
        hidden, memory = hidden

        num_directions = 2 if self.bidirectional else 1
        batch_size = hidden.size(1)

        # Separate the layers from the number of directions
        # shape: (num_layers, num_directions, batch_size, hidden_size)
        hidden = hidden.view(self.num_layers, num_directions, batch_size, self.hidden_size)
        # shape: (num_layers, num_directions, batch_size, hidden_size)
        memory = memory.view(self.num_layers, num_directions, batch_size, self.hidden_size)

        # If this is uni-directional, then we can remove the directions
        # dimension and return
        if num_directions == 1:
            # shape: (1, batch_size, hidden_size)
            hidden = hidden.squeeze(0)
            # shape: (1, batch_size, hidden_size)
            memory = memory.squeeze(0)
            return hidden, memory
        else:
            # Otherwise, we have to concatenate the two directions into one vector
            # shape: (num_layers, batch_size, hidden_size * 2)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            # shape: (num_layers, batch_size, hidden_size * 2)
            memory = torch.cat([memory[:, 0, :, :], memory[:, 1, :, :]], dim=2)
            return hidden, memory
