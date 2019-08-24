"""
Computes the document-frequency term for calculating BM25. The model will consider
the cloze context as the query and the reference document sentences as the
documents that need to be ranked. Therefore, one sentence is a "document" in the
BM25 equation, and thus the document frequencies should be based on the document
sentences.
"""
import argparse
from collections import Counter
from tqdm import tqdm

from summarize.data.io import JsonlReader, JsonlWriter


def main(args):
    dfs = Counter()
    total_document_length = 0
    num_documents = 0

    with JsonlReader(args.input_jsonl) as f:
        for instance in tqdm(f, desc='Calculating document frequencies'):
            document = instance['document']
            for sentence in document:
                tokens = sentence.lower().split()
                total_document_length += len(tokens)
                num_documents += 1
                for token in set(tokens):
                    dfs[token] += 1

    average_document_length = total_document_length / num_documents
    with JsonlWriter(args.output_jsonl) as out:
        out.write({'num_documents': num_documents, 'average_document_length': average_document_length})
        for token, df in dfs.items():
            out.write({'token': token, 'df': df})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl')
    argp.add_argument('output_jsonl')
    args = argp.parse_args()
    main(args)
