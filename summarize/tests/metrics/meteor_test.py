import unittest

from summarize.metrics.meteor import run_meteor


class TestMeteor(unittest.TestCase):
    def test_meteor_runs(self):
        gold_summaries = [
            'This is the gold summary for the first instance.',
            'And this is for the second one.'
        ]
        model_summaries = [
            'This is the model output.',
            'And this is the one for the second document.'
        ]
        assert run_meteor(gold_summaries, model_summaries) > 0.0
