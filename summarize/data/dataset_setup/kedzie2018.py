"""
Prepares the datasets to reproduce Kedzie 2018 by computing greedy oracle
summaries by optimizing ROUGE-1 recall.
"""
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, List

from summarize.data.io import JsonlReader, JsonlWriter
from summarize.metrics.python_rouge import PythonRouge
from summarize.metrics.rouge import R1_RECALL
from summarize.models.sds.oracle import get_greedy_oracle_summary

_BATCH_SIZE = 100


def _process_batch(parallel: Parallel,
                   batch: List[Dict[str, List[str]]],
                   max_tokens: int,
                   python_rouge: PythonRouge,
                   out: JsonlWriter) -> None:
    jobs = []
    for instance in batch:
        document = instance['document']
        summary = instance['summary']
        job = delayed(get_greedy_oracle_summary)(document, summary,
                                                 R1_RECALL,
                                                 max_tokens=max_tokens,
                                                 use_porter_stemmer=True,
                                                 remove_stopwords=True,
                                                 python_rouge=python_rouge)
        jobs.append(job)

    results = parallel(jobs)
    for instance, (_, labels) in zip(batch, results):
        instance['labels'] = labels
        out.write(instance)


def main(args):
    python_rouge = PythonRouge()
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            with Parallel(n_jobs=args.num_cores) as parallel:
                batch = []
                for instance in tqdm(f):
                    batch.append(instance)
                    if len(batch) == _BATCH_SIZE:
                        _process_batch(parallel, batch, args.max_tokens, python_rouge, out)
                        batch.clear()

                if batch:
                    _process_batch(parallel, batch, args.max_tokens, python_rouge, out)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The dataset to setup')
    argp.add_argument('output_jsonl', help='The output file')
    argp.add_argument('max_tokens', type=int, help='The maximum number of tokens to take in the greedy summary')
    argp.add_argument('--num-cores', type=int, default=1, help='The number of cores to use')
    args = argp.parse_args()
    main(args)
