"""
Computes the lead baseline for single-document summarization.
"""
import argparse
from typing import List, Optional

from summarize.data.io import JsonlReader, JsonlWriter


def get_lead_summary(document: List[str],
                     max_sentences: Optional[int] = None,
                     max_tokens: Optional[int] = None,
                     max_bytes: Optional[int] = None) -> List[str]:
    """
    Gets the lead summary of the input document. Exactly one of ``max_sentences``,
    ``max_tokens``, and ``max_bytes`` must be non-None.

    Parameters
    ----------
    document:
        The sentence-tokenized input document to summarize. If using ``max_tokens``,
        the document should already tokenized and separated with whitespace.
    max_sentences:
        The maximum number of lead sentences to take.
    max_tokens:
        The maximum number of tokens to take.
    max_bytes:
        The maximum number of bytes to take, where each character is counted
        as a byte (including whitespace).

    Returns
    -------
    The lead summary with the original sentence boundaries maintained.
    """
    args = [max_sentences, max_tokens, max_bytes]
    if sum([arg is not None for arg in args]) != 1:
        raise Exception('Exactly one of `max_sentences`, `max_tokens`, and `max_bytes` must be set.')

    if max_sentences is not None:
        return document[:max_sentences]
    elif max_tokens is not None:
        budget = max_tokens
        summary = []
        for sentence in document:
            tokens = sentence.split()[:budget]
            summary.append(' '.join(tokens))
            budget -= len(tokens)
            if budget <= 0:
                break
        return summary
    elif max_bytes is not None:
        budget = max_bytes
        summary = []
        for sentence in document:
            sentence = sentence[:budget].strip()
            summary.append(sentence)
            budget -= len(sentence) + 1  # the space between sentences
            if budget <= 0:
                break
        return summary


def main(args):
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in f:
                document = instance['document']
                summary = get_lead_summary(document,
                                           max_sentences=args.max_sentences,
                                           max_tokens=args.max_tokens,
                                           max_bytes=args.max_bytes)
                out.write({'summary': summary})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The input documents')
    argp.add_argument('output_jsonl', help='The output file')
    argp.add_argument('--max-sentences', type=int, help='The maximum number of sentences to take.')
    argp.add_argument('--max-tokens', type=int, help='The maximum number of tokens to take.')
    argp.add_argument('--max-bytes', type=int, help='The maximum number of bytes to take.')
    args = argp.parse_args()
    main(args)
