import torch
from overrides import overrides

from summarize.nn.beam_search.coverage_penalizers import CoveragePenalizer


@CoveragePenalizer.register('onmt')
class ONMTCoveragePenalizer(CoveragePenalizer):
    """
    An implementation of the "summary" coverage penalty in the OpenNMT machine
    translation library (https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/penalties.py).
    Because we add the coverage penalty to the log-probabilies (as in Wu et al.),
    instead of subtracting (as in ONMT), the sign of this penalty is the opposite
    as the ONMT implementation.

    The penalty discourages the coverage from attending to any one token too often.

    Parameters
    ----------
    beta: ``float``
        The scaling factor.
    """
    def __init__(self, beta: float) -> None:
        self.beta = beta

    @overrides
    def __call__(self, coverage: torch.Tensor) -> torch.Tensor:
        num_document_tokens = coverage.size(-1)
        penalty = num_document_tokens - torch.clamp(coverage, 1.0).sum(dim=-1)
        return self.beta * penalty
