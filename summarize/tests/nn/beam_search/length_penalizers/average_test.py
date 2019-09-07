import torch
import unittest

from summarize.nn.beam_search.length_penalizers import AverageLengthPenalizer


class TestAverageLengthPenalizer(unittest.TestCase):
    def test_average_length_penalizer(self):
        lengths = torch.LongTensor([[1, 2], [3, 4]])

        penalizer = AverageLengthPenalizer()
        penalties = penalizer(lengths)
        assert torch.equal(lengths.float(), penalties)
