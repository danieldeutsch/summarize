import torch
from allennlp.common.registrable import Registrable
from typing import Tuple


class CoverageMatrixAttention(torch.nn.Module, Registrable):
    """
    The ``CoverageMatrixAttention`` computes a matrix of attention probabilities
    between the encoder and decoder outputs. The attention function has access
    to the cumulative probabilities that the attention has assigned to each
    input token previously. In addition to the attention probabilities, the function
    should return the coverage vectors which were used to compute the distribution
    at each time step as well as the new coverage vector which takes into account
    the function's computation.

    The module must compute the probabilities instead of the raw scores (like
    the ``MatrixAttention`` module does) because the coverage vector contains
    the accumulated probabilities.
    """
    def forward(self,
                decoder_outputs: torch.Tensor,
                encoder_outputs: torch.Tensor,
                encoder_mask: torch.Tensor,
                coverage_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes a matrix of attention scores and updates the coverage vector.

        Parameters
        ----------
        decoder_outputs: (batch_size, num_decoder_tokens, hidden_dim)
            The decoder's outputs.
        encoder_outputs: (batch_size, num_encoder_tokens, hidden_dim)
            The encoder's outputs.
        encoder_mask: (batch_size, num_encoder_tokens)
            The encoder token mask.
        coverage_vector: (batch_size, num_encoder_tokens)
            The cumulative attention probability assigned to each input token
            thus far.

        Returns
        -------
        torch.Tensor: (batch_size, num_decoder_tokens, num_encoder_tokens)
            The attention probabilities between each decoder and encoder hidden representations.
        torch.Tensor: (batch_size, num_decoder_tokens, num_encoder_tokens)
            The coverage vectors used to compute the corresponding attention probabilities.
        torch.Tensor: (batch_size, num_encoder_tokens)
            The latest coverage vector after computing
        """
        raise NotImplementedError
