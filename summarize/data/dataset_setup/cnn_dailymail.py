"""
Prepares the CNN/DailyMail dataset. This script uses the original data from
https://cs.nyu.edu/~kcho/DMQA/ and preprocesses the data with modifications of
the script from See et al. (2017) (https://github.com/abisee/cnn-dailymail).
The output data has the original text, not normalized or tokenized.
"""
import argparse
import hashlib
import os
import tarfile
from io import BytesIO
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from summarize.data.dataset_setup import util
from summarize.data.io import JsonlWriter

_dm_single_close_quote = u'\u2019'  # unicode
_dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', _dm_single_close_quote, _dm_double_close_quote, ")"]


def get_url_hash(url: str) -> str:
    def hashhex(string: str) -> str:
        h = hashlib.sha1()
        h.update(string)
        return h.hexdigest()
    return hashhex(url.encode("utf8"))


def fix_missing_period(line):
    if '@highlight' in line:
        return line
    if line == '':
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + '.'


def parse_story(story_bytes: bytes) -> Tuple[List[str], List[str]]:
    # Load the story lines from the bytes
    lines = [line.decode().strip() for line in story_bytes]

    # Add potentially missing periods
    lines = [fix_missing_period(line) for line in lines]

    # Parse the article and the highlights
    article, highlights = [], []
    next_is_highlight = False
    for line in lines:
        if line == '':
            continue  # empty line
        elif line.startswith('@highlight'):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article.append(line)

    # The lines do not perfectly align with sentence boundaries, so try to fix that
    article = [sentence for line in article for sentence in sent_tokenize(line)]
    highlights = [sentence for line in highlights for sentence in sent_tokenize(line)]
    return article, highlights


def load_instances(tgz_path: str, corpus_name: str) -> List[Dict[str, Any]]:
    instances = {}
    with tarfile.open(tgz_path, 'r:gz') as tar:
        for member in tqdm(tar.getmembers(), desc=f'Processing {tgz_path}'):
            if member.name.endswith('.story'):
                story_bytes = BytesIO(tar.extractfile(member.name).read())
                article, highlights = parse_story(story_bytes)
                if len(article) == 0 or len(highlights) == 0:
                    continue

                filename = os.path.basename(member.name)
                hash_ = filename.rstrip('.story')
                instances[hash_] = {
                    'filename': filename,
                    'document': article,
                    'summary': highlights
                }
    return instances


def save_data(instances: Dict[str, Any], urls: List[str], file_path: str) -> None:
    with JsonlWriter(file_path) as out:
        for url in tqdm(urls, desc=f'Saving instances to {file_path}'):
            hash_ = get_url_hash(url)
            if hash_ in instances:
                instance = instances[hash_]
                instance['url'] = url
                out.write(instance)


def main(args):
    # Download and load the url splits
    train_url_file_path = f'{args.output_dir}/train-urls.txt'
    valid_url_file_path = f'{args.output_dir}/valid-urls.txt'
    test_url_file_path = f'{args.output_dir}/test-urls.txt'
    util.download_url_to_file('https://github.com/abisee/cnn-dailymail/raw/master/url_lists/all_train.txt', train_url_file_path)
    util.download_url_to_file('https://github.com/abisee/cnn-dailymail/raw/master/url_lists/all_val.txt', valid_url_file_path)
    util.download_url_to_file('https://github.com/abisee/cnn-dailymail/raw/master/url_lists/all_test.txt', test_url_file_path)

    train_urls = open(train_url_file_path, 'r').read().splitlines()
    valid_urls = open(valid_url_file_path, 'r').read().splitlines()
    test_urls = open(test_url_file_path, 'r').read().splitlines()

    # These are the stories files from https://cs.nyu.edu/~kcho/DMQA/
    cnn_tgz = f'{args.output_dir}/cnn.tgz'
    dailymail_tgz = f'{args.output_dir}/dailymail.tgz'
    util.download_from_google_drive('0BwmD_VLjROrfTHk4NFg2SndKcjQ', 158577824, cnn_tgz)
    util.download_from_google_drive('0BwmD_VLjROrfM1BxdkxVaTY2bWs', 375893739, dailymail_tgz)

    cnn_instances = load_instances(cnn_tgz, 'cnn')
    dailymail_instances = load_instances(dailymail_tgz, 'dailymail')
    all_instances = {**cnn_instances, **dailymail_instances}

    cnn_train_path = f'{args.output_dir}/cnn/train.jsonl.gz'
    cnn_valid_path = f'{args.output_dir}/cnn/valid.jsonl.gz'
    cnn_test_path = f'{args.output_dir}/cnn/test.jsonl.gz'
    save_data(cnn_instances, train_urls, cnn_train_path)
    save_data(cnn_instances, valid_urls, cnn_valid_path)
    save_data(cnn_instances, test_urls, cnn_test_path)

    dailymail_train_path = f'{args.output_dir}/dailymail/train.jsonl.gz'
    dailymail_valid_path = f'{args.output_dir}/dailymail/valid.jsonl.gz'
    dailymail_test_path = f'{args.output_dir}/dailymail/test.jsonl.gz'
    save_data(dailymail_instances, train_urls, dailymail_train_path)
    save_data(dailymail_instances, valid_urls, dailymail_valid_path)
    save_data(dailymail_instances, test_urls, dailymail_test_path)

    all_train_path = f'{args.output_dir}/cnn-dailymail/train.jsonl.gz'
    all_valid_path = f'{args.output_dir}/cnn-dailymail/valid.jsonl.gz'
    all_test_path = f'{args.output_dir}/cnn-dailymail/test.jsonl.gz'
    save_data(all_instances, train_urls, all_train_path)
    save_data(all_instances, valid_urls, all_valid_path)
    save_data(all_instances, test_urls, all_test_path)

    util.assert_line_count(all_train_path, 287113)
    util.assert_line_count(all_valid_path, 13368)
    util.assert_line_count(all_test_path, 11490)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('output_dir', help='The directory where the CNN/DailyMail data should be saved')
    args = argp.parse_args()
    main(args)
