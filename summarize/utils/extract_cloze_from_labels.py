import argparse

from summarize.data.io import JsonlReader, JsonlWriter


def main(args):
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in f:
                document = instance['document']
                labels = instance['labels']
                cloze = [document[index] for index in labels]
                if not args.keep_sentences:
                    cloze = ' '.join(cloze)
                out.write({args.field_name: cloze})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The input file with the labeled summaries.')
    argp.add_argument('output_jsonl', help='The output file')
    argp.add_argument('--field-name', default='cloze', help='The name of the output field')
    argp.add_argument('--keep-sentences', action='store_true', help='Indicates if the output field should be left as sentences or flattened')
    args = argp.parse_args()
    main(args)
