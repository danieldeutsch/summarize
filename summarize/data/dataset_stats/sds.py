"""
Computes basic statistics about single-document summarization datasets.
"""
import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from summarize.data.io import JsonlReader

NUM_DOC_TOKENS = 'num-document-tokens'
NUM_DOC_SENTS = 'num-document-sentences'
NUM_SUM_TOKENS = 'num-summary-tokens'
NUM_SUM_SENTS = 'num-summary-sentences'
NUM_INSTANCES = 'num-instances'

AVG_DOC_SENT_TOKENS = 'avg-document-sentence-tokens'
AVG_DOC_SENTS = 'avg-document-sentences'
AVG_SUM_SENT_TOKENS = 'avg-summary-sentence-tokens'
AVG_SUM_SENTS = 'avg-summary-sentences'


def get_default_stats() -> Dict[str, float]:
    return {
        NUM_DOC_TOKENS: [],
        NUM_DOC_SENTS: [],
        NUM_SUM_TOKENS: [],
        NUM_SUM_SENTS: [],
        NUM_INSTANCES: 0
    }


def compute_file_statistics(file_path: str) -> Dict[str, float]:
    stats = get_default_stats()
    with JsonlReader(file_path) as f:
        for instance in tqdm(f, desc=f'Processing {file_path}'):
            document = instance['document']
            summary = instance['summary']
            stats[NUM_DOC_TOKENS] += [sum(len(sentence.split()) for sentence in document)]
            stats[NUM_DOC_SENTS] += [len(document)]
            stats[NUM_SUM_TOKENS] += [sum(len(sentence.split()) for sentence in summary)]
            stats[NUM_SUM_SENTS] += [len(summary)]
            stats[NUM_INSTANCES] += 1
    return stats


def combine_statistics(*stats_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    combined = get_default_stats()
    for stats in stats_dicts:
        for metric in [NUM_DOC_TOKENS, NUM_DOC_SENTS, NUM_SUM_TOKENS, NUM_SUM_SENTS, NUM_INSTANCES]:
            combined[metric] += stats[metric]
    return combined


def compute_final_statistics(stats: Dict[str, float]) -> Dict[str, float]:
    avg_doc_sents = np.average(stats[NUM_DOC_SENTS])
    std_doc_sents = np.std(stats[NUM_DOC_SENTS])
    avg_doc_sent_tokens = np.average(np.array(stats[NUM_DOC_TOKENS]) / np.array(stats[NUM_DOC_SENTS]))
    std_doc_sent_tokens = np.std(np.array(stats[NUM_DOC_TOKENS]) / np.array(stats[NUM_DOC_SENTS]))
    avg_sum_sents = np.average(stats[NUM_SUM_SENTS])
    std_sum_sents = np.std(stats[NUM_SUM_SENTS])
    avg_sum_sent_tokens = np.average(np.array(stats[NUM_SUM_TOKENS]) / np.array(stats[NUM_SUM_SENTS]))
    std_sum_sent_tokens = np.std(np.array(stats[NUM_SUM_TOKENS]) / np.array(stats[NUM_SUM_SENTS]))
    return {
        NUM_INSTANCES: stats[NUM_INSTANCES],
        AVG_DOC_SENTS: f'{avg_doc_sents:.2f} ({std_doc_sents:.2f})',
        AVG_DOC_SENT_TOKENS: f'{avg_doc_sent_tokens:.2f} ({std_doc_sent_tokens:.2f})',
        AVG_SUM_SENTS: f'{avg_sum_sents:.2f} ({std_sum_sents:.2f})',
        AVG_SUM_SENT_TOKENS: f'{avg_sum_sent_tokens:.2f} ({std_sum_sent_tokens:.2f})',
    }


def main(args):
    train_stats = compute_file_statistics(args.train_file_path)
    valid_stats = compute_file_statistics(args.valid_file_path)
    test_stats = compute_file_statistics(args.test_file_path)
    combined_stats = combine_statistics(train_stats, valid_stats, test_stats)

    final_train_stats = compute_final_statistics(train_stats)
    final_valid_stats = compute_final_statistics(valid_stats)
    final_test_stats = compute_final_statistics(test_stats)
    final_combined_stats = compute_final_statistics(combined_stats)

    print('Train statistics')
    print(json.dumps(final_train_stats, indent=4))
    print()
    print('Valid statistics')
    print(json.dumps(final_valid_stats, indent=4))
    print()
    print('Test statistics')
    print(json.dumps(final_test_stats, indent=4))
    print()
    print('Combined statistics')
    print(json.dumps(final_combined_stats, indent=4))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('train_file_path', help='The path to the training jsonl file')
    argp.add_argument('valid_file_path', help='The path to the validation jsonl file')
    argp.add_argument('test_file_path', help='The path to the testing jsonl file')
    args = argp.parse_args()
    main(args)
