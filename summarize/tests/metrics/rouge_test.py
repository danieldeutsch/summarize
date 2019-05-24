import json
import os
import pytest
import unittest
from typing import List

from summarize.common.testing import FIXTURES_ROOT
from summarize.metrics.rouge import run_rouge, R1_RECALL, R2_RECALL, R4_RECALL

_duc2004_file_path = 'data/duc/duc2004/duc2004.task2.jsonl'
_centroid_file_path = f'{FIXTURES_ROOT}/data/hong2014/centroid.jsonl'
_classy04_file_path = f'{FIXTURES_ROOT}/data/hong2014/classy04.jsonl'
_classy11_file_path = f'{FIXTURES_ROOT}/data/hong2014/classy11.jsonl'
_dpp_file_path = f'{FIXTURES_ROOT}/data/hong2014/dpp.jsonl'
_freq_sum_file_path = f'{FIXTURES_ROOT}/data/hong2014/freq-sum.jsonl'
_greedy_kl_file_path = f'{FIXTURES_ROOT}/data/hong2014/greedy-kl.jsonl'
_icsi_summ_file_path = f'{FIXTURES_ROOT}/data/hong2014/icsi-summ.jsonl'
_lexrank_file_path = f'{FIXTURES_ROOT}/data/hong2014/lexrank.jsonl'
_occams_v_file_path = f'{FIXTURES_ROOT}/data/hong2014/occams-v.jsonl'
_reg_sum_file_path = f'{FIXTURES_ROOT}/data/hong2014/reg-sum.jsonl'
_submodular_file_path = f'{FIXTURES_ROOT}/data/hong2014/submodular.jsonl'
_ts_sum_file_path = f'{FIXTURES_ROOT}/data/hong2014/ts-sum.jsonl'


class TestRouge(unittest.TestCase):
    def _load_summaries(self, file_path: str) -> List[List[str]]:
        summaries = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                summaries.append(data['summary'])
        return summaries

    def _load_multiple_summaries(self, file_path: str) -> List[List[List[str]]]:
        summaries = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                summaries.append(data['summaries'])
        return summaries

    @pytest.mark.skipif(not os.path.exists(_duc2004_file_path), reason='DUC 2004 data does not exist')
    def test_hong2014(self):
        """
        Tests to ensure that the Rouge scores for the summaries from Hong et al. 2014
        (http://www.lrec-conf.org/proceedings/lrec2014/pdf/1093_Paper.pdf) do not
        change. The hard-coded scores are very close to the scores reported in the paper.
        """
        duc2004 = self._load_multiple_summaries(_duc2004_file_path)
        centroid = self._load_summaries(_centroid_file_path)
        classy04 = self._load_summaries(_classy04_file_path)
        classy11 = self._load_summaries(_classy11_file_path)
        dpp = self._load_summaries(_dpp_file_path)
        freq_sum = self._load_summaries(_freq_sum_file_path)
        greedy_kl = self._load_summaries(_greedy_kl_file_path)
        icsi_summ = self._load_summaries(_icsi_summ_file_path)
        lexrank = self._load_summaries(_lexrank_file_path)
        occams_v = self._load_summaries(_occams_v_file_path)
        reg_sum = self._load_summaries(_reg_sum_file_path)
        submodular = self._load_summaries(_submodular_file_path)
        ts_sum = self._load_summaries(_ts_sum_file_path)

        # Reported: 36.41, 7.97, 1.21
        metrics = run_rouge(duc2004, centroid, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 36.41, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 7.97, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.21, places=2)

        # Reported: 37.62, 8.96, 1.51
        metrics = run_rouge(duc2004, classy04, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 37.61, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 8.96, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.51, places=2)

        # Reported: 37.22, 9.20, 1.48
        metrics = run_rouge(duc2004, classy11, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 37.22, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 9.20, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.48, places=2)

        # Reported: 39.79, 9.62, 1.57
        metrics = run_rouge(duc2004, dpp, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 39.79, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 9.62, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.57, places=2)

        # Reported: 35.30, 8.11, 1.00
        metrics = run_rouge(duc2004, freq_sum, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 35.30, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 8.11, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.00, places=2)

        # Reported: 37.98, 8.53, 1.26
        metrics = run_rouge(duc2004, greedy_kl, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 37.98, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 8.53, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.26, places=2)

        # Reported: 38.41, 9.78, 1.73
        metrics = run_rouge(duc2004, icsi_summ, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 38.41, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 9.78, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.73, places=2)

        # Reported: 35.95, 7.47, 0.82
        metrics = run_rouge(duc2004, lexrank, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 35.95, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 7.47, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 0.82, places=2)

        # Reported: 38.50, 9.76, 1.33
        metrics = run_rouge(duc2004, occams_v, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 38.50, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 9.76, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.33, places=2)

        # Reported: 38.57, 9.75, 1.60
        metrics = run_rouge(duc2004, reg_sum, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 38.56, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 9.75, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.60, places=2)

        # Reported: 39.18, 9.35, 1.39
        metrics = run_rouge(duc2004, submodular, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 39.18, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 9.35, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.39, places=2)

        # Reported: 35.88, 8.15, 1.03
        metrics = run_rouge(duc2004, ts_sum, max_words=100)
        self.assertAlmostEqual(metrics[R1_RECALL], 35.88, places=2)
        self.assertAlmostEqual(metrics[R2_RECALL], 8.14, places=2)
        self.assertAlmostEqual(metrics[R4_RECALL], 1.03, places=2)
