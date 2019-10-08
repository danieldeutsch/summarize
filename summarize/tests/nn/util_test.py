import torch
import unittest

from summarize.nn.util import normalize_losses


class TestUtil(unittest.TestCase):
    def test_normalize_losses(self):
        losses = torch.FloatTensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        mask = torch.FloatTensor([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0]
        ])

        actual_loss = normalize_losses(losses, mask, 'sum', 'sum')
        expected_loss = 15.0
        assert expected_loss == actual_loss.item()

        actual_loss = normalize_losses(losses, mask, 'sum', 'average')
        expected_loss = 7.5
        assert expected_loss == actual_loss.item()

        actual_loss = normalize_losses(losses, mask, 'average', 'sum')
        expected_loss = 6.5
        assert expected_loss == actual_loss.item()

        actual_loss = normalize_losses(losses, mask, 'average', 'average')
        expected_loss = 3.25
        assert expected_loss == actual_loss.item()

        with self.assertRaises(Exception):
            normalize_losses(losses, mask, 'unknown', 'sum')
        with self.assertRaises(Exception):
            normalize_losses(losses, mask, 'sum', 'unknown')
