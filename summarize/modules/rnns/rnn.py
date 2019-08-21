import torch
from allennlp.common import Registrable
from overrides import overrides
from typing import Optional, Tuple, Union

from summarize.modules.rnns import util


class RNN(torch.nn.Module, Registrable):
    """
    The ``RNN`` is an abstraction over different types of recurrent neural networks.
    It is similar in spirit to AllenNLP's ``Seq2SeqEncoder`` or ``Seq2VecEncoder``, but
    it has more control over the hidden state of the RNNs. Specifically, the AllenNLP
    abstractions only have access to the final hidden state by using
    ``allennlp.nn.util.get_final_encoder_states`` function. This is fine for GRUs, but
    not for LSTMs which also have memory cells, which are not accessible through
    AllenNLP. The default behavior for when the memory cell is not provided to initialize
    the decoder is to set it to all 0's, but in our experiments, we observed that
    initializing the decoder's memory cell to the encoder's final memory cell actually
    made a difference in performance, and so we created this abstraction.

    In this abstraction, the ``forward`` method returns both the outputs and the
    RNN's hidden state instead of just the outputs. For the GRU, the hidden state
    will just be a single tensor, but for the LSTM, the hidden state will be a tuple
    of two tensors. The ``RNN`` class handles the logic which is common among
    the RNN classes.

    Parameters
    ----------
    input_size: ``int``, required
        The size of the input dimension.
    hidden_size: ``hidden_size``, required
        The size of the hidden dimension.
    num_layers: ``num_layers``, required
        The number of layers.
    bidirectional: ``bool``, required
        Indicates if the RNN is bidirectional or not.
    rnn: ``torch.nn.RNNBase``, required
        The underlying RNN module. The RNN should always be batch first.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 rnn: torch.nn.RNNBase) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn = rnn

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor],
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Runs the RNN over the inputs.

        Parameters
        ----------
        inputs: ``torch.Tensor``, ``(batch_size, sequence_length, input_size)``
            The input vectors for the RNN.
        mask: ``torch.Tensor``, ``(batch_size, sequence_length)``, required
            The input mask. If ``None``, no masking will be used.
        hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]``, ``(num_layers * num_directions, batch_size, hidden_size)``, optional
            The hidden state of the RNN. Each tensor should be the above size. If ``None``, the
            PyTorch default behavior for the RNN will be used.
        """
        if mask is None:
            outputs, hidden = self.rnn(inputs, hidden)
        else:
            packed_inputs, packed_hidden = util.pack_sequence(inputs, mask, hidden)
            packed_outputs, packed_hidden = self.rnn(packed_inputs, packed_hidden)
            outputs, hidden = util.unpack_sequence(packed_outputs, packed_hidden, mask)
        return outputs, hidden

    def get_input_dim(self) -> int:
        return self.input_size

    def get_output_dim(self):
        return (int(self.bidirectional) + 1) * self.hidden_size

    def is_bidirectional(self) -> bool:
        return self.bidirectional

    def has_memory(self) -> bool:
        """Indicates whether or not this RNN has a memory cell (i.e., LSTMs)."""
        raise NotImplementedError

    def reshape_hidden_for_decoder(self, hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reshapes the hidden state of the RNN so it can be used to initialize
        a decoder RNN.

        Parameters
        ----------
        hidden: ``Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]``, ``(num_layers * num_directions, batch_size, hidden_size)``
            The final hidden state of the encoder.

        Returns
        -------
        The ``(num_layers, batch_size, hidden_size * num_directions)`` reshaped tensors.
        """
        raise NotImplementedError
