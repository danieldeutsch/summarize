"""
Ranks the document sentences according to BM25. The model will consider
the cloze context as the query and the reference document sentences as the
documents that need to be ranked. Therefore, one sentence is a "document" in the
BM25 equation.
"""
import argparse
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from typing import Dict, List, Set, Tuple

from summarize.data.io import JsonlReader, JsonlWriter


def load_dfs(file_path: str) -> Tuple[Dict[str, int], int, float]:
    dfs = defaultdict(int)
    with JsonlReader(file_path) as f:
        for i, data in enumerate(f):
            if i == 0:
                num_documents = data['num_documents']
                avg_document_length = data['average_document_length']
            else:
                token = data['token']
                df = data['df']
                dfs[token] = df
    return dfs, num_documents, avg_document_length


def calculate_bm25(query_tokens: Set[str],
                   document: List[str],
                   dfs: Dict[str, int],
                   num_documents: int,
                   avg_document_length: float,
                   k: float,
                   b: float) -> float:
    bm25 = 0.0
    tf = Counter(document)
    for token in query_tokens:
        idf = np.log((num_documents + dfs[token] + 0.5) / (dfs[token] + 0.5))
        bm25 += idf * (tf[token] * (k + 1)) / (tf[token] + k * (1 - b + b * len(document) / avg_document_length))
    return bm25


def get_cloze(bm25_scores: List[Tuple[float, str]], max_words: int, max_sents: int, flatten: bool):
    if max_words is None and max_sents is None:
        raise Exception('At least one of `max_words` and `max_sents` must be set.')

    # Sort by BM25 score descending
    bm25_scores.sort(key=lambda t: -t[0])

    cloze = []
    num_words = 0
    for _, sentence in bm25_scores:
        cloze.append(sentence)
        if max_words is not None:
            num_words += len(sentence.split())
            if num_words >= max_words:
                break

        if max_sents is not None and len(cloze) >= max_sents:
            break

    if flatten:
        cloze = ' '.join(cloze)
    return cloze


def main(args):
    dfs, num_documents, avg_document_length = load_dfs(args.df_jsonl)

    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in tqdm(f):
                context = instance['context']
                context_tokens = set(token.lower() for sentence in context for token in sentence.split())
                document = instance['document']

                bm25_scores = []
                for sentence in document:
                    tokenized_sentence = [token.lower() for token in sentence.split()]
                    bm25 = calculate_bm25(context_tokens, tokenized_sentence,
                                          dfs, num_documents, avg_document_length,
                                          args.k, args.b)

                    bm25_scores.append((bm25, sentence))

                cloze = get_cloze(bm25_scores, args.max_words, args.max_sentences, args.flatten)
                out.write({'cloze': cloze})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl')
    argp.add_argument('df_jsonl')
    argp.add_argument('output_jsonl')
    argp.add_argument('--k', type=float, default=1.2)
    argp.add_argument('--b', type=float, default=0.75)
    argp.add_argument('--max-words', type=int)
    argp.add_argument('--max-sentences', type=int)
    argp.add_argument('--flatten', action='store_true')
    args = argp.parse_args()
    main(args)
