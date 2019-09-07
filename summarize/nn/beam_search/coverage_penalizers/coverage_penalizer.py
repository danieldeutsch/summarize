"""
``CoveragePenalizer``s are used to rerank the output of beam search by adding
a penalty to the score of each prediction at each step of decoding.
"""
import torch

from allennlp.common import Registrable


class CoveragePenalizer(Registrable):
    def __call__(self, coverage: torch.Tensor) -> torch.Tensor:
        """
        Computes the factor that should be added to the log-probability of
        each output step.

        Parameters
        ----------
        coverage: ``torch.Tensor``, (..., num_document_tokens)
            A tensor that represents the accumulated attention probabilities
            assigned to each document token thus far in decoding. The tensor
            may have any number of leading dimensions.

        Returns
        -------
        ``torch.Tensor``:
            A tensor with the coverage penalties, the same size as the leading
            dimensions as the coverage tensor.
        """
        raise NotImplementedError
