import numpy as np
import torch
from overrides import overrides

from summarize.nn.beam_search.length_penalizers import LengthPenalizer


@LengthPenalizer.register('wu')
class WuLengthPenalizer(LengthPenalizer):
    """
    Implements the length penalty in Wu et al. (2016) (https://arxiv.org/pdf/1609.08144.pdf),
    section 7.

    Parameters
    ----------
    alpha: ``float``
        The value of alpha in the length penalty.
    """
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    @overrides
    def __call__(self, length: torch.Tensor) -> torch.Tensor:
        return torch.pow(5.0 + length.float(), self.alpha) / np.power(6.0, self.alpha)
