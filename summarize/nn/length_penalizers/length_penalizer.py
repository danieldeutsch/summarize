"""
``LengthPenalizer``s are used to rerank the output of beam search. After all
the top-k hypotheses have been found, their log-probability scores are divided
by a length penalty to adjust for different lengths.
"""
import torch

from allennlp.common import Registrable


class LengthPenalizer(Registrable):
    def __call__(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Computes the factor that the log-probability of the output sequence
        should be divded by based on its length.

        Parameters
        ----------
        lengths: ``torch.Tensor``
            A tensor of the lengths, which can be any size.

        Returns
        -------
        ``torch.Tensor``:
            A tensor with the length penalties, the same size as the input tensor.
        """
        raise NotImplementedError
