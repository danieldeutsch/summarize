import torch
from overrides import overrides

from summarize.nn.beam_search.length_penalizers import LengthPenalizer


@LengthPenalizer.register('average')
class AverageLengthPenalizer(LengthPenalizer):
    """
    Penalizes by predictions length of the sequence, thus causing the score
    to be the average log-probability per token.
    """
    @overrides
    def __call__(self, length: torch.Tensor) -> torch.Tensor:
        return length.float()
