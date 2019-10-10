import os
import pytest
import unittest

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.io import JsonlReader
from summarize.metrics.meteor import DEFAULT_METEOR_JAR_PATH, run_meteor


@pytest.mark.skipif(not os.path.exists(DEFAULT_METEOR_JAR_PATH), reason='Meteor jar does not exist')
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

    def test_chen2018(self):
        """
        Tests to ensure that Meteor returns the expected score on the
        Chen 2018 data subset. I ran Meteor on the full data (~11k examples)
        which takes too long to run for a unit test. After confirming the numbers
        are the same as what is reported in the paper, I ran the code on just
        the subset, and this test ensures those numbers are returned.
        """
        gold_file_path = f'{FIXTURES_ROOT}/data/chen2018/gold.jsonl'
        model_file_path = f'{FIXTURES_ROOT}/data/chen2018/model.jsonl'

        gold = JsonlReader(gold_file_path).read()
        model = JsonlReader(model_file_path).read()

        gold = [' '.join(summary['summary']) for summary in gold]
        model = [' '.join(summary['summary']) for summary in model]

        score = run_meteor(gold, model)
        assert abs(score - 18.28372) < 1e-5
