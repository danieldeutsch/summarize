# flake8: noqa
import argparse
import sys
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.append('../summarize')

from summarize.data.io import JsonlReader


def main(args):
    # The number of times each document appears
    document_to_num_occurrences = Counter()
    # The histogram of the document set sizes
    document_set_sizes = Counter()
    # The mapping from the document to the page ids
    document_to_page_ids = defaultdict(set)

    with JsonlReader(args.input_jsonl) as f:
        for instance in tqdm(f):
            page_id = instance['page_id']
            documents = instance['documents']
            document_set_sizes[len(documents)] += 1

            for document in documents:
                url = document['canonical_url']
                document_to_num_occurrences[url] += 1
                document_to_page_ids[url].add(page_id)

    # The histogram for the number of times a document appears
    num_occurrences_to_num_documents = Counter()
    for count in document_to_num_occurrences.values():
        num_occurrences_to_num_documents[count] += 1

    # The histogram for the number of pages a document appears
    num_pages_to_num_documents = Counter()
    for page_ids in document_to_page_ids.values():
        num_pages_to_num_documents[len(page_ids)] += 1

    num_instances = sum(document_set_sizes.values())
    num_multidoc = num_instances - document_set_sizes[1]

    num_unique_documents = len(document_to_num_occurrences)
    num_documents_multiple_times = num_unique_documents - num_occurrences_to_num_documents[1]

    num_documents_multiple_pages = num_unique_documents - num_pages_to_num_documents[1]

    print(f'Total unique documents: {num_unique_documents}')
    print(f'Total multi-document: {num_multidoc} ({num_multidoc / num_instances * 100:.2f}%)')
    print(f'Total documents appear more than once: {num_documents_multiple_times} ({num_documents_multiple_times / num_unique_documents * 100:.2f}%)')
    print(f'Total documents that appear in more than one page: {num_documents_multiple_pages} ({num_documents_multiple_pages / num_unique_documents * 100:.2f}%)')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The WikiCite dataset to analyze')
    args = argp.parse_args()
    main(args)
