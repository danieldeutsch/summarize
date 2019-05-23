"""
Downloads and reformats the Gigaword dataset from
https://github.com/harvardnlp/sent-summary
"""
import argparse
import gzip
import os
import tarfile
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from io import BytesIO
from tqdm import tqdm
from typing import Dict, List

from summarize.data.dataset_setup import util
from summarize.data.io import JsonlWriter


def replace_unk_with_oov(line: str) -> str:
    """
    The original Gigaword data has "UNK" tokens to represent unknown words.
    This function will replace UNK with the AllenNLP special token for OOV.
    """
    tokens = line.split()
    for i, token in enumerate(tokens):
        if token == 'UNK':
            tokens[i] = DEFAULT_OOV_TOKEN
    return ' '.join(tokens)


def load_data(gigaword_tar: str, article_path: str, title_path: str) -> List[Dict[str, List[str]]]:
    instances = []
    with tarfile.open(gigaword_tar, 'r:gz') as tar:
        article_file = tar.getmember(article_path)
        title_file = tar.getmember(title_path)

        article_bytes = BytesIO(tar.extractfile(article_file).read())
        title_bytes = BytesIO(tar.extractfile(title_file).read())

        if article_path.endswith('.gz'):
            article_f = gzip.open(article_bytes, 'rb')
        else:
            article_f = article_bytes
        if title_path.endswith('.gz'):
            title_f = gzip.open(title_bytes, 'rb')
        else:
            title_f = title_bytes

        for article_line, title_line in tqdm(zip(article_f, title_f)):
            article_line = article_line.decode().strip()
            title_line = title_line.decode().strip()

            article_line = replace_unk_with_oov(article_line)
            title_line = replace_unk_with_oov(title_line)

            instance = {
                'document': [article_line],
                'summary': [title_line]
            }
            instances.append(instance)

    return instances


def load_training_data(gigaword_tar: str) -> List[Dict[str, List[str]]]:
    return load_data(gigaword_tar, 'sumdata/train/train.article.txt.gz',
                     'sumdata/train/train.title.txt.gz')


def load_valid_data(gigaword_tar: str) -> List[Dict[str, List[str]]]:
    return load_data(gigaword_tar, 'sumdata/train/valid.article.filter.txt',
                     'sumdata/train/valid.title.filter.txt')


def load_test_data(gigaword_tar: str) -> List[Dict[str, List[str]]]:
    return load_data(gigaword_tar, 'sumdata/Giga/input.txt',
                     'sumdata/Giga/task1_ref0.txt')


def save_data(data: List[Dict[str, List[str]]], file_path: str) -> None:
    with JsonlWriter(file_path) as out:
        for item in tqdm(data, desc=f'Writing instances to {file_path}'):
            out.write(item)


def main(args):
    # Downloads the Gigaword dataset as preprocessed here:
    # https://github.com/harvardnlp/sent-summary
    gigaword_tar = os.path.join(args.output_dir, 'summary.tar.gz')
    util.download_from_google_drive('0B6N7tANPyVeBNmlSX19Ld2xDU1E', 290866023, gigaword_tar)

    train = load_training_data(gigaword_tar)
    valid = load_valid_data(gigaword_tar)
    test = load_test_data(gigaword_tar)

    train_file_path = os.path.join(args.output_dir, 'train.jsonl.gz')
    valid_file_path = os.path.join(args.output_dir, 'valid.jsonl.gz')
    test_file_path = os.path.join(args.output_dir, 'test.jsonl.gz')
    save_data(train, train_file_path)
    save_data(valid, valid_file_path)
    save_data(test, test_file_path)

    util.assert_line_count(train_file_path, 3803957)
    util.assert_line_count(valid_file_path, 189651)
    util.assert_line_count(test_file_path, 1951)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('output_dir', help='The directory where the Gigaword data should be saved')
    args = argp.parse_args()
    main(args)
