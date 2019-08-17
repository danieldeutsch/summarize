import torch
import unittest

from summarize.training.metrics import BinaryF1Measure


class BinaryF1MeasureTest(unittest.TestCase):
    def test_binary_f1_measure(self):
        gold_labels = torch.LongTensor([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0]
        ])
        model_labels = torch.LongTensor([
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])
        mask = torch.LongTensor([
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0]
        ])

        metric = BinaryF1Measure()
        expected_precision = 3 / 5
        expected_recall = 3 / 4
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)

        metric(gold_labels, model_labels, mask)
        actual_metrics = metric.get_metric()
        self.assertAlmostEqual(actual_metrics['precision'], expected_precision, delta=1e-5)
        self.assertAlmostEqual(actual_metrics['recall'], expected_recall, delta=1e-5)
        self.assertAlmostEqual(actual_metrics['f1'], expected_f1, delta=1e-5)
