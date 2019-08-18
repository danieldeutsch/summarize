"""
Preprocesses the original WikiCite dataset by tokenizing all of the text fields.
"""
import argparse
import spacy
from tqdm import tqdm

from summarize.data.dataset_setup.tokenize import tokenize
from summarize.data.io import JsonlReader, JsonlWriter


def main(args):
    nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in tqdm(f):
                instance['headings'] = [tokenize(nlp, heading) for heading in instance['headings']]
                for document in instance['documents']:
                    if document['title']:
                        document['title'] = tokenize(nlp, document['title'])
                    document['paragraphs'] = tokenize(nlp, document['paragraphs'])

                instance['left_context'] = tokenize(nlp, instance['left_context'])
                instance['cloze'] = tokenize(nlp, instance['cloze'])
                instance['right_context'] = tokenize(nlp, instance['right_context'])
                out.write(instance)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The input file to setup')
    argp.add_argument('output_jsonl', help='The output file')
    args = argp.parse_args()
    main(args)
