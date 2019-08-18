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
                   python_rouge: PythonRouge,
                   out: JsonlWriter) -> None:
    jobs = []
    documents = []
    for instance in batch:
        document = [sentence for document in instance['documents']
                    for paragraph in document['paragraphs']
                    for sentence in paragraph]
        cloze = instance['cloze']
        job = delayed(get_greedy_oracle_summary)(document, [cloze], R1_RECALL,
                                                 use_porter_stemmer=True,
                                                 remove_stopwords=False,
                                                 python_rouge=python_rouge)
        jobs.append(job)
        documents.append(document)

    results = parallel(jobs)
    for instance, document, (_, labels) in zip(batch, documents, results):
        id_ = instance['id']
        page_title = instance['page_title']
        headings = instance['headings']
        topics = [page_title] + headings
        context = instance['left_context']
        cloze = instance['cloze']
        output_data = {
            'id': id_,
            'topics': topics,
            'document': document,
            'context': context,
            'cloze': cloze,
            'labels': labels
        }
        out.write(output_data)


def main(args):
    python_rouge = PythonRouge()
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            with Parallel(n_jobs=args.num_cores) as parallel:
                batch = []
                for instance in tqdm(f):
                    batch.append(instance)
                    if len(batch) == _BATCH_SIZE:
                        _process_batch(parallel, batch, python_rouge, out)
                        batch.clear()

                if batch:
                    _process_batch(parallel, batch, python_rouge, out)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The input file to preprocess')
    argp.add_argument('output_jsonl', help='The output file')
    argp.add_argument('--num-cores', type=int, default=1, help='The number of cores to use')
    args = argp.parse_args()
    main(args)
