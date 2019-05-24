"""
This is a python reimplementation of ROUGE. The official perl script is very slow
and requires disk I/O. This script aims to be as close as possible to the original
perl implementation. However, it should only be used as an approximation for the
official ROUGE scores. For final experimentation, the ``rouge.py`` metric should
be used instead of this one.
"""
import argparse
import json
import os
import re
from collections import Counter
from nltk.stem import PorterStemmer
from typing import Dict, List, Optional, Set, Tuple, Union

from summarize.data.io import JsonlReader
from summarize.metrics.rouge import R1_RECALL, R1_PRECISION, R1_F1, \
    R2_RECALL, R2_PRECISION, R2_F1, \
    R3_RECALL, R3_PRECISION, R3_F1, \
    R4_RECALL, R4_PRECISION, R4_F1, \
    RL_RECALL, RL_PRECISION, RL_F1
from summarize.metrics.rouge import has_multiple_references

_non_alphanumeric_regex = re.compile('[^A-Za-z0-9]')


def _load_summaries(file_path: str, field_name: str = 'summary') -> Union[List[List[str]], List[List[List[str]]]]:
    summaries = []
    with JsonlReader(file_path) as f:
        for data in f:
            summaries.append(data[field_name])
    return summaries


def shorten_summary(summary: List[str],
                    max_sentences: Optional[int] = None,
                    max_words: Optional[int] = None,
                    max_bytes: Optional[int] = None) -> List[str]:
    args = [max_sentences, max_words, max_bytes]
    if sum(1 if arg is not None else 0 for arg in args) not in [0, 1]:
        raise Exception(f'Only one of `max_sentences`, `max_words`, and `max_bytes` can be set.')

    shortened_summary = []
    if max_sentences is not None:
        shortened_summary = summary[:max_sentences]
    elif max_words is not None:
        budget = max_words
        for sentence in summary:
            tokens = sentence.split()[:budget]
            shortened_summary.append(' '.join(tokens))
            budget -= len(tokens)
            if budget <= 0:
                break
    elif max_bytes is not None:
        budget = max_bytes
        for sentence in summary:
            sentence = sentence[:budget].strip()
            shortened_summary.append(sentence)
            budget -= len(sentence)
            if budget <= 0:
                break
    else:
        shortened_summary = summary

    return shortened_summary


class PythonRouge(object):
    """
    The ``PythonRouge`` class is the python implementation of the official ROUGE
    perl script. The goal is to be as close as possible to the official computation,
    however, it is very likely there are some minor differences. The implementation
    was made into a class because the class may be called many times in a row
    (for example, to compute heuristic extractive labels by greedily maximizing
    ROUGE), and it would be too expensive to load the external dependencies every time.

    If you are computing the official ROUGE scores for evaluation, you should
    use the ``rouge.py`` script instead.

    Parameters
    ----------
    data_dir: ``str``, optional (default = "external/ROUGE-1.5.5/data")
        The path to the ROUGE data directory.
    """
    def __init__(self, data_dir: str = 'external/ROUGE-1.5.5/data') -> None:
        self.stemmer = PorterStemmer(PorterStemmer.ORIGINAL_ALGORITHM)
        self.stemmer_exceptions = self._load_stemmer_exceptions(data_dir)
        self.stopwords = self._load_stopwords(data_dir)

    def _load_stemmer_exceptions(self, root: str) -> Dict[str, str]:
        exceptions = {}
        for filename in ['adj.exc', 'adv.exc', 'noun.exc', 'verb.exc']:
            file_path = os.path.join(root, 'WordNet-2.0-Exceptions', filename)
            with open(file_path, 'r') as f:
                for line in f:
                    # I think there is a bug in the original perl script
                    # to construct the exceptions database. Some of the lines
                    # have more than 2 words on them, but the script only
                    # maps the first to the second, ignoring the third.
                    columns = line.strip().split()
                    exceptions[columns[0]] = columns[1]
        return exceptions

    def _load_stopwords(self, root: str) -> Set[str]:
        file_path = os.path.join(root, 'smart_common_words.txt')
        return set(open(file_path, 'r').read().splitlines())

    def normalize_and_tokenize_sentence(self,
                                        sentence: str,
                                        use_porter_stemmer: bool = True,
                                        remove_stopwords: bool = False) -> List[str]:
        sentence = _non_alphanumeric_regex.sub(' ', sentence)
        sentence = sentence.lower()
        tokens = []
        for token in sentence.split():
            if remove_stopwords and token in self.stopwords:
                continue
            if use_porter_stemmer and len(token) > 3:
                if token in self.stemmer_exceptions:
                    tokens.append(self.stemmer_exceptions[token])
                else:
                    tokens.append(self.stemmer.stem(token))
            else:
                tokens.append(token)
        return tokens

    def _normalize_and_tokenize_summary(self,
                                        summary: List[str],
                                        use_porter_stemmer: bool = True,
                                        remove_stopwords: bool = False) -> List[str]:
        normalized_summary = []
        for sentence in summary:
            sentence = self.normalize_and_tokenize_sentence(sentence, use_porter_stemmer, remove_stopwords)
            normalized_summary.append(sentence)
        return normalized_summary

    def preprocess_summary(self,
                           summary: str,
                           max_sentences: Optional[int] = None,
                           max_words: Optional[int] = None,
                           max_bytes: Optional[int] = None,
                           use_porter_stemmer: bool = True,
                           remove_stopwords: bool = False) -> List[List[str]]:
        summary = shorten_summary(summary, max_sentences, max_words, max_bytes)
        summary = self._normalize_and_tokenize_summary(summary, use_porter_stemmer, remove_stopwords)
        return summary

    def _preprocess_summaries(self,
                              summaries_list: List[List[List[str]]],
                              max_sentences: Optional[int] = None,
                              max_words: Optional[int] = None,
                              max_bytes: Optional[int] = None,
                              use_porter_stemmer: bool = True,
                              remove_stopwords: bool = False):
        preprocessed_summaries_list = []
        for summaries in summaries_list:
            preprocessed_summaries = []
            for summary in summaries:
                preprocessed_summary = self.preprocess_summary(summary, max_sentences, max_words,
                                                               max_bytes, use_porter_stemmer,
                                                               remove_stopwords)
                preprocessed_summaries.append(preprocessed_summary)
            preprocessed_summaries_list.append(preprocessed_summaries)
        return preprocessed_summaries_list

    def _count_ngrams(self, sentences: List[List[str]], n: int) -> Counter:
        counts = Counter()
        tokens = [token for sentence in sentences for token in sentence]
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            counts[ngram] += 1
        return counts

    def _calculate_pr_f1(self, gold_total: int, model_total: int, intersection: int) -> Tuple[float, float, float]:
        precision = 0.0
        if model_total != 0.0:
            precision = intersection / model_total
        recall = 0.0
        if gold_total != 0.0:
            recall = intersection / gold_total
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    def _calculate_intersection(self, gold_counts: Counter, model_counts: Counter) -> Tuple[float, float, float]:
        gold_total = sum(gold_counts.values())
        model_total = sum(model_counts.values())
        intersection = 0
        for ngram in model_counts:
            intersection += min(model_counts[ngram], gold_counts[ngram])
        return gold_total, model_total, intersection

    def _calculate_rouge(self,
                         gold_summaries: List[List[List[str]]],
                         model_summary: List[List[str]],
                         ngram_order: int):
        total_gold_count, total_model_count, total_intersection = 0, 0, 0

        model_ngrams = self._count_ngrams(model_summary, ngram_order)
        for gold_summary in gold_summaries:
            gold_ngrams = self._count_ngrams(gold_summary, ngram_order)

            gold_total, model_total, intersection = self._calculate_intersection(gold_ngrams, model_ngrams)
            total_gold_count += gold_total
            total_model_count += model_total
            total_intersection += intersection

        precision, recall, f1 = self._calculate_pr_f1(total_gold_count, total_model_count, total_intersection)
        return precision, recall, f1

    def _longest_common_substring(self,
                                  tokens1: List[str],
                                  tokens2: List[str],
                                  hit_mask: List[int]) -> int:
        m, n = len(tokens1), len(tokens2)
        counter = [[0] * (n + 1) for x in range(m + 1)]
        pointers = [[None] * (n + 1) for x in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    counter[i][j] = counter[i - 1][j - 1] + 1
                    pointers[i][j] = '\\'
                elif counter[i - 1][j] >= counter[i][j - 1]:
                    counter[i][j] = counter[i - 1][j]
                    pointers[i][j] = '^'
                else:
                    counter[i][j] = counter[i][j - 1]
                    pointers[i][j] = '<'

        # Mark the hit_mask
        i, j = m, n
        while i != 0 and j != 0:
            if pointers[i][j] == '\\':
                i -= 1
                j -= 1
                hit_mask[i] = 1
            elif pointers[i][j] == '^':
                i -= 1
            elif pointers[i][j] == '<':
                j -= 1
            else:
                raise Exception(f'Unknown pointer: {pointers[i][j]}')

    def _calculate_rouge_l(self,
                           gold_summaries: List[List[List[str]]],
                           model_summary: List[List[str]]):
        model_unigrams = self._count_ngrams(model_summary, 1)
        num_model_unigrams = sum(count for count in model_unigrams.values())

        total_hit = 0
        total_base = 0
        for gold_summary in gold_summaries:
            temp_model_unigrams = Counter(model_unigrams)
            gold_unigrams = self._count_ngrams(gold_summary, 1)
            hit, base = 0, 0
            for gold_sentence in gold_summary:
                hit_mask = [0] * len(gold_sentence)
                base += len(gold_sentence)
                for model_sentence in model_summary:
                    self._longest_common_substring(gold_sentence, model_sentence, hit_mask)

                for i, token in enumerate(gold_sentence):
                    if hit_mask[i] == 1:
                        try:
                            if temp_model_unigrams[token] > 0 and gold_unigrams[token] > 0:
                                hit += 1
                                temp_model_unigrams[token] -= 1
                                gold_unigrams[token] -= 1
                        except KeyError:
                            pass
            total_hit += hit
            total_base += base

        precision = 0.0
        if (num_model_unigrams * len(gold_summaries)) != 0.0:
            precision = total_hit / (num_model_unigrams * len(gold_summaries))
        recall = 0.0
        if total_base != 0.0:
            recall = total_hit / total_base
        if (precision + recall) != 0.0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return precision, recall, f1

    def run_python_rouge(self,
                         gold_summaries: Union[List[List[str]], List[List[List[str]]]],
                         model_summaries: List[List[str]],
                         ngram_orders: List[int] = None,
                         max_sentences: Optional[int] = None,
                         max_words: Optional[int] = None,
                         max_bytes: Optional[int] = None,
                         use_porter_stemmer: bool = True,
                         remove_stopwords: bool = False,
                         compute_rouge_l: bool = False):
        """
        Runs the python implementation of ROUGE. Each individual summary should be
        represented as a ``List[str]``.

        Parameters
        ----------
        gold_summaries: ``Union[List[List[str]], List[List[List[str]]]]``
            The ground-truth summaries. If there is only one ground-truth summary per
            instance, the type should be ``List[List[str]]``. If there are multiple, then
            ``List[List[List[str]]]``.
        model_summaries: ``List[List[str]]``
            The model summaries to evaluate.
        ngram_orders: ``List[int]``, optional (default = ``[1, 2]``)
            The n-gram orders that ROUGE should be computed for.
        max_sentences: ``int``, optional (default = `None`)
            Limits the length of each summary by a maximum number of sentences. If
            ``None``, no truncation is performed. This option cannot be used with
            ``max_words`` or ``max_bytes``.
        max_words: ``int``, optional (default = `None`)
            Limits the length of each summary by a maximum number of words. If
            ``None``, no truncation is performed. This option cannot be used with
            ``max_sentences`` or ``max_bytes``.
        max_bytes: ``int``, optional (default = `None`)
            Limits the length of each summary by a maximum number of bytes. If
            ``None``, no truncation is performed. This option cannot be used with
            ``max_sentences`` or ``max_words``.
        use_porter_stemmer: ``bool``, optional (default = ``True``)
            Indicates whether or not the summaries should be preprocessed using
            the Porter Stemmer.
        remove_stopwords: ``bool``, optional (default = ``False``)
            Indicates whether stopwords should be removed from the summaries.
        compute_rouge_l:  ``bool``, optional (default = ``False``)
            Indicates whether ROUGE-L should be computed.

        Returns
        -------
        A dictionary that maps from the metric name to the value.
        """
        multiple_references = has_multiple_references(gold_summaries)
        if multiple_references:
            gold_summaries_list = gold_summaries
        else:
            gold_summaries_list = [[gold_summary] for gold_summary in gold_summaries]
        if ngram_orders is None:
            ngram_orders = [1, 2]
        if not ngram_orders and not compute_rouge_l:
            raise Exception('At least one n-gram order or Rouge-L must be given')

        gold_summaries_list = self._preprocess_summaries(gold_summaries_list, max_sentences, max_words, max_bytes,
                                                         use_porter_stemmer, remove_stopwords)
        model_summaries = self._preprocess_summaries([model_summaries], max_sentences, max_words, max_bytes,
                                                     use_porter_stemmer, remove_stopwords)[0]

        metrics = {}
        for ngram_order in ngram_orders:
            total_precision, total_recall, total_f1 = 0, 0, 0
            for gold_summaries, model_summary in zip(gold_summaries_list, model_summaries):
                precision, recall, f1 = self._calculate_rouge(gold_summaries, model_summary, ngram_order)
                total_precision += precision
                total_recall += recall
                total_f1 += f1

            precision = total_precision / len(gold_summaries_list) * 100
            recall = total_recall / len(gold_summaries_list) * 100
            f1 = total_f1 / len(gold_summaries_list) * 100

            if ngram_order == 1:
                metrics[R1_PRECISION] = precision
                metrics[R1_RECALL] = recall
                metrics[R1_F1] = f1
            elif ngram_order == 2:
                metrics[R2_PRECISION] = precision
                metrics[R2_RECALL] = recall
                metrics[R2_F1] = f1
            elif ngram_order == 3:
                metrics[R3_PRECISION] = precision
                metrics[R3_RECALL] = recall
                metrics[R3_F1] = f1
            elif ngram_order == 4:
                metrics[R4_PRECISION] = precision
                metrics[R4_RECALL] = recall
                metrics[R4_F1] = f1
            else:
                raise Exception(f'Unsupported ngram order: {ngram_order}')

        if compute_rouge_l:
            total_precision, total_recall, total_f1 = 0, 0, 0
            for gold_summaries, model_summary in zip(gold_summaries_list, model_summaries):
                precision, recall, f1 = self._calculate_rouge_l(gold_summaries, model_summary)
                total_precision += precision
                total_recall += recall
                total_f1 += f1

            precision = total_precision / len(gold_summaries_list) * 100
            recall = total_recall / len(gold_summaries_list) * 100
            f1 = total_f1 / len(gold_summaries_list) * 100

            metrics[RL_PRECISION] = precision
            metrics[RL_RECALL] = recall
            metrics[RL_F1] = f1

        return metrics


def main(args):
    if args.silent and not args.output_file:
        raise Exception(f'No output will be written with --silent and no output file')

    gold_summaries = _load_summaries(args.gold_summaries, args.gold_summary_field_name)
    model_summaries = _load_summaries(args.model_summaries)

    rouge = PythonRouge()
    metrics = rouge.run_python_rouge(gold_summaries, model_summaries)

    if not args.silent:
        print(json.dumps(metrics, indent=4))
    if args.output_file:
        with open(args.output_file, 'w') as out:
            out.write(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('gold_summaries', help='The path to the gold summary jsonl file.')
    argp.add_argument('model_summaries', help='The path to the model summary jsonl file.')
    argp.add_argument('--output-file',
                      help='The path to the file where the json output should be written.')
    argp.add_argument('--gold-summary-field-name', default='summary',
                      help='The field name in the gold json for the summaries (e.g., "summary" for '
                           'single reference or "summaries" for multi-reference).')
    argp.add_argument('--silent', action='store_true',
                      help='Indicates nothing should be written to stdout.')
    args = argp.parse_args()
    main(args)
