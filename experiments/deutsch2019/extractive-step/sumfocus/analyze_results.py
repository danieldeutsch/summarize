import argparse
import json
import os
from collections import defaultdict
from glob import glob
from typing import Dict, Tuple


def get_hyperparameters(filename: str) -> Tuple[float, float, float]:
    # valid.beta_2.topic-lambda-0.3.context-lambda-0.4.metrics.json
    beta_index = filename.find('beta')
    topic_index = filename.find('topic')
    context_index = filename.find('context')
    metrics_index = filename.find('metrics.json')
    beta = float(filename[beta_index + 5:topic_index - 1])
    topic_lambda = float(filename[topic_index + 13:context_index - 1])
    context_lambda = float(filename[context_index + 15:metrics_index - 1])
    return beta, topic_lambda, context_lambda


def get_best_document(results) -> Dict[str, float]:
    best_setting = None
    best_value = None
    for beta, topic_lambda_dict in results.items():
        if 0.0 in topic_lambda_dict:
            context_lambda_dict = topic_lambda_dict[0.0]
            if 0.0 in context_lambda_dict:
                metrics = context_lambda_dict[0.0]
                value = metrics['R2-R']
                if best_value is None or value > best_value:
                    best_value = value
                    best_setting = (beta, 0.0, 0.0)
    return best_setting


def get_best_document_topic(results) -> Dict[str, float]:
    best_setting = None
    best_value = None
    for beta, topic_lambda_dict in results.items():
        for topic_lambda, context_lambda_dict in topic_lambda_dict.items():
            if 0.0 in context_lambda_dict:
                metrics = context_lambda_dict[0.0]
                value = metrics['R2-R']
                if best_value is None or value > best_value:
                    best_value = value
                    best_setting = (beta, topic_lambda, 0.0)
    return best_setting


def get_best_document_context(results) -> Dict[str, float]:
    best_setting = None
    best_value = None
    for beta, topic_lambda_dict in results.items():
        if 0.0 in topic_lambda_dict:
            context_lambda_dict = topic_lambda_dict[0.0]
            for context_lambda, metrics in context_lambda_dict.items():
                value = metrics['R2-R']
                if best_value is None or value > best_value:
                    best_value = value
                    best_setting = (beta, 0.0, context_lambda)
    return best_setting


def get_best_document_topic_context(results) -> Dict[str, float]:
    best_setting = None
    best_value = None
    for beta, topic_lambda_dict in results.items():
        for topic_lambda, context_lambda_dict in topic_lambda_dict.items():
            for context_lambda, metrics in context_lambda_dict.items():
                value = metrics['R2-R']
                if best_value is None or value > best_value:
                    best_value = value
                    best_setting = (beta, topic_lambda, context_lambda)
    return best_setting


def main(args):
    results_dir = args.results_dir

    results = defaultdict(lambda: defaultdict(dict))
    for file_path in glob(f'{results_dir}/*.metrics.json'):
        filename = os.path.basename(file_path)
        beta, topic_lambda, context_lambda = get_hyperparameters(filename)
        metrics = json.loads(open(file_path, 'r').read())
        results[beta][topic_lambda][context_lambda] = metrics

    print('SumFocused-Topic-Context')
    beta, topic_lambda, context_lambda = get_best_document(results)
    metrics = results[beta][topic_lambda][context_lambda]
    print(f'beta = {beta}')
    print(f'topic_lambda = {topic_lambda}')
    print(f'context_lambda = {context_lambda}')
    print(json.dumps(metrics, indent=4))
    print()

    print('SumFocused-Context')
    beta, topic_lambda, context_lambda = get_best_document_topic(results)
    metrics = results[beta][topic_lambda][context_lambda]
    print(f'beta = {beta}')
    print(f'topic_lambda = {topic_lambda}')
    print(f'context_lambda = {context_lambda}')
    print(json.dumps(metrics, indent=4))
    print()

    print('SumFocused-Topic')
    beta, topic_lambda, context_lambda = get_best_document_context(results)
    metrics = results[beta][topic_lambda][context_lambda]
    print(f'beta = {beta}')
    print(f'topic_lambda = {topic_lambda}')
    print(f'context_lambda = {context_lambda}')
    print(json.dumps(metrics, indent=4))
    print()

    print('SumFocused')
    beta, topic_lambda, context_lambda = get_best_document_topic_context(results)
    metrics = results[beta][topic_lambda][context_lambda]
    print(f'beta = {beta}')
    print(f'topic_lambda = {topic_lambda}')
    print(f'context_lambda = {context_lambda}')
    print(json.dumps(metrics, indent=4))
    print()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('results_dir')
    args = argp.parse_args()
    main(args)
