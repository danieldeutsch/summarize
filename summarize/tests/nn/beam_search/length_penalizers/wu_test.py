import torch
import unittest

from summarize.nn.beam_search.length_penalizers import WuLengthPenalizer


class TestWuLengthPenalizer(unittest.TestCase):
    def test_wu_length_penalizer(self):
        lengths = torch.LongTensor([[1, 2], [3, 4]])

        penalizer = WuLengthPenalizer(0.0)
        penalties = penalizer(lengths)
        expected_penalties = torch.FloatTensor([[1.0, 1.0], [1.0, 1.0]])
        assert torch.allclose(expected_penalties, penalties)

        penalizer = WuLengthPenalizer(0.5)
        penalties = penalizer(lengths)
        expected_penalties = torch.FloatTensor([[1.0, 1.0801], [1.1547, 1.2247]])
        assert torch.allclose(expected_penalties, penalties, atol=1e-3)
