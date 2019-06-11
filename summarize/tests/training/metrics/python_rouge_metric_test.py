import os
import pytest
import torch
import unittest
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, Vocabulary
from typing import Dict, List

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.io import JsonlReader
from summarize.metrics.python_rouge import PythonRouge
from summarize.training.metrics import PythonRougeMetric

_duc2004_file_path = 'data/duc/duc2004/task2.jsonl'
_centroid_file_path = f'{FIXTURES_ROOT}/data/hong2014/centroid.jsonl'


@pytest.mark.skipif(not os.path.exists(_duc2004_file_path), reason='DUC 2004 data does not exist')
class PythonRougeMetricTest(unittest.TestCase):
    def _load_gold_summaries(self) -> List[List[str]]:
        # Loads just the first summary for testing purposes
        summaries = []
        with JsonlReader(_duc2004_file_path) as f:
            for instance in f:
                summaries.append(instance['summaries'][0])
        return summaries

    def _load_model_summaries(self) -> List[List[str]]:
        summaries = []
        with JsonlReader(_centroid_file_path) as f:
            for instance in f:
                summaries.append(instance['summary'])
        return summaries

    def _build_vocabulary(self, summaries: List[List[str]]):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(START_SYMBOL)
        vocab.add_token_to_namespace(END_SYMBOL)
        for summary in summaries:
            for sentence in summary:
                for token in sentence.split():
                    vocab.add_token_to_namespace(token)
        return vocab

    def _flatten_summaries(self, summaries: List[List[str]]) -> List[List[str]]:
        flattened_summaries = []
        for summary in summaries:
            flattened_summaries.append([' '.join(summary)])
        return flattened_summaries

    def _convert_to_tensors(self,
                            summaries: List[List[str]],
                            batch_size: int,
                            vocab: Vocabulary) -> List[torch.Tensor]:
        tensors = []
        for i in range(0, len(summaries), batch_size):
            tensor = self._convert_to_tensor(summaries[i:i + batch_size], vocab)
            tensors.append(tensor)
        return tensors

    def _convert_to_tensor(self, summaries: List[List[str]], vocab: Vocabulary) -> torch.Tensor:
        # Adds <bos> and <eos> just for testing purposes
        tokenized_summaries = [[sentence.split() for sentence in summary] for summary in summaries]
        batch_size = len(tokenized_summaries)
        num_sents = max(len(summary) for summary in tokenized_summaries)
        num_tokens = max(len(tokens) for summary in tokenized_summaries for tokens in summary) + 2

        pad_index = vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        tensor = torch.zeros(batch_size, num_sents, num_tokens).fill_(pad_index)
        for batch, summary in enumerate(tokenized_summaries):
            for i, tokens in enumerate(summary):
                for j, token in enumerate([START_SYMBOL] + tokens + [END_SYMBOL]):
                    tensor[batch, i, j] = vocab.get_token_index(token)
        return tensor

    def setUp(self):
        self.gold_summaries = self._load_gold_summaries()
        self.model_summaries = self._load_model_summaries()
        self.vocab = self._build_vocabulary([*self.gold_summaries, *self.model_summaries])
        self.gold_summaries_tensors = self._convert_to_tensors(self.gold_summaries, 2, self.vocab)
        self.model_summaries_tensors = self._convert_to_tensors(self.model_summaries, 2, self.vocab)

        self.gold_summaries_abs = self._flatten_summaries(self.gold_summaries)
        self.model_summaries_abs = self._flatten_summaries(self.model_summaries)
        self.gold_summaries_tensors_abs = self._convert_to_tensors(self.gold_summaries_abs, 2, self.vocab)
        self.model_summaries_tensors_abs = self._convert_to_tensors(self.model_summaries_abs, 2, self.vocab)

    def _assert_metrics_equal(self,
                              expected_metrics: Dict[str, float],
                              actual_metrics: Dict[str, float]) -> None:
        assert len(expected_metrics) == len(actual_metrics)
        for metric in expected_metrics.keys():
            self.assertAlmostEqual(expected_metrics[metric], actual_metrics[metric], delta=1e-5)

    def test_python_rouge_metric_extractive(self):
        """
        Tests to ensure that the `PythonRougeMetric` will compute the same Rouge
        scores as the `PythonRouge` class for extractive summaries.
        """
        ngram_orders = [1, 2]
        max_words = 100

        python_rouge = PythonRouge()
        metric = PythonRougeMetric(vocab=self.vocab, ngram_orders=ngram_orders, max_words=max_words)

        expected_metrics = python_rouge.run_python_rouge(self.gold_summaries,
                                                         self.model_summaries,
                                                         ngram_orders=ngram_orders,
                                                         max_words=max_words)

        # Test passing batched strings
        batch_size = 2
        for i in range(0, len(self.gold_summaries), batch_size):
            metric(self.gold_summaries[i:i + batch_size], self.model_summaries[i:i + batch_size])
        actual_metrics = metric.get_metric(reset=True)
        self._assert_metrics_equal(expected_metrics, actual_metrics)

        # Test passing batched tensors
        for gold_tensor, model_tensor in zip(self.gold_summaries_tensors, self.model_summaries_tensors):
            metric(gold_tensor, model_tensor)
        actual_metrics = metric.get_metric(reset=True)
        self._assert_metrics_equal(expected_metrics, actual_metrics)

        # Test passing tensors batched with lists
        for gold_tensor, model_tensor in zip(self.gold_summaries_tensors, self.model_summaries_tensors):
            gold_list = [tensor for tensor in gold_tensor]
            model_list = [tensor for tensor in model_tensor]
            metric(gold_list, model_list)
        actual_metrics = metric.get_metric(reset=True)
        self._assert_metrics_equal(expected_metrics, actual_metrics)

    def test_python_rouge_metric_abstractive(self):
        """
        Tests to ensure that the `PythonRougeMetric` will compute the same Rouge
        scores as the `PythonRouge` class for abstractive summaries (that aren't
        sentence tokenized).
        """
        ngram_orders = [1, 2]
        max_words = 100
        use_stemmer = False
        remove_stopwords = True

        python_rouge = PythonRouge()
        metric = PythonRougeMetric(vocab=self.vocab, ngram_orders=ngram_orders,
                                   max_words=max_words, use_porter_stemmer=use_stemmer,
                                   remove_stopwords=remove_stopwords)

        expected_metrics = python_rouge.run_python_rouge(self.gold_summaries_abs,
                                                         self.model_summaries_abs,
                                                         ngram_orders=ngram_orders,
                                                         max_words=max_words,
                                                         use_porter_stemmer=use_stemmer,
                                                         remove_stopwords=remove_stopwords)

        # Test passing batched strings
        batch_size = 2
        for i in range(0, len(self.gold_summaries_abs), batch_size):
            metric(self.gold_summaries_abs[i:i + batch_size], self.model_summaries_abs[i:i + batch_size])
        actual_metrics = metric.get_metric(reset=True)
        self._assert_metrics_equal(expected_metrics, actual_metrics)

        # Test passing batched tensors
        for gold_tensor, model_tensor in zip(self.gold_summaries_tensors_abs, self.model_summaries_tensors_abs):
            metric(gold_tensor, model_tensor)
        actual_metrics = metric.get_metric(reset=True)
        self._assert_metrics_equal(expected_metrics, actual_metrics)

        # Test passing tensors batched with lists
        for gold_tensor, model_tensor in zip(self.gold_summaries_tensors_abs, self.model_summaries_tensors_abs):
            gold_list = [tensor for tensor in gold_tensor]
            model_list = [tensor for tensor in model_tensor]
            metric(gold_list, model_list)
        actual_metrics = metric.get_metric(reset=True)
        self._assert_metrics_equal(expected_metrics, actual_metrics)
