"""
Tokenizes fields in a jsonl dataset file with the English spacy tokenizer.
"""
import argparse
import nltk
import spacy
from tqdm import tqdm
from typing import Callable, Iterable, T

from summarize.data.io import JsonlReader, JsonlWriter


def tokenize(tokenize_func: Callable[[str], Iterable[T]], field):
    """
    Tokenizes text using the a tokenizer function. The ``field`` argument can be
    a string or a nested list of strings. The method will return the same level of nesting
    with the tokens whitespace separated in a string.

    The ``tokenize_func`` should be some function which returns iterable of tokens
    which can be cast to strings. For example, the ``nlp`` object from spacy or
    the ``word_tokenize`` function from nltk both work.

    For example::

        nlp = spacy.load('en')
        tokenize(nlp, "Hi, I'm Dan.")
        >>> "Hi , I 'm Dan ."
        tokenize(nlp, [['The first.', 'The second.'], 'The third.'])
        >>> [['The first .', 'The second .'], 'The third .']

        from nltk import word_tokenize
        tokenize(word_tokenize, 'This is the NLTK version.')
        >>> 'This is the NLTK version .'

    Parameters
    ----------
    tokenize_func: ``Callable[[str], Iterable[T]]``, required.
        The tokenization function. See above for a more detailed explanation.
    field: required.
        The text to tokenize. See above for the type explanation.

    Returns
    -------
    The tokenized text.
    """
    if isinstance(field, str):
        return ' '.join([str(token) for token in tokenize_func(field)])
    elif isinstance(field, list):
        return [tokenize(tokenize_func, item) for item in field]
    else:
        raise TypeError(f'Unknown ``field`` type {type(field)}')


def main(args):
    if args.backend == 'spacy':
        nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])
    elif args.backend == 'nltk':
        nlp = nltk.word_tokenize

    with JsonlWriter(args.output_file) as out:
        with JsonlReader(args.input_file) as f:
            for instance in tqdm(f, desc=f'Tokenizing {args.input_file}'):
                for field in args.fields:
                    instance[field] = tokenize(nlp, instance[field])
                out.write(instance)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_file', help='The jsonl file with fields to tokenize')
    argp.add_argument('output_file', help='The output jsonl file with the tokenized data')
    argp.add_argument('fields', nargs='+')
    argp.add_argument('--backend', default='spacy', choices=['spacy', 'nltk'],
                      help='Indicates which library should be used for tokenization')
    args = argp.parse_args()
    main(args)
