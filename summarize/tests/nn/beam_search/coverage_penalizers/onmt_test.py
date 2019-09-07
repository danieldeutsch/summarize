import torch
import unittest

from summarize.nn.beam_search.coverage_penalizers import ONMTCoveragePenalizer


class TestAverageLengthPenalizer(unittest.TestCase):
    def test_onmt_coverage_penalizer(self):
        coverage = torch.FloatTensor([[0.4, 1.2, 0.8], [1.5, 0.7, 0.0]])

        penalizer = ONMTCoveragePenalizer(0.0)
        penalties = penalizer(coverage)
        expected_penalties = torch.FloatTensor([0.0, 0.0])
        assert torch.allclose(expected_penalties, penalties)

        penalizer = ONMTCoveragePenalizer(0.5)
        penalties = penalizer(coverage)
        expected_penalties = torch.FloatTensor([-0.2 * 0.5, -0.5 * 0.5])
        assert torch.allclose(expected_penalties, penalties, atol=1e-3)
