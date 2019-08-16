import argparse

from summarize.data.io import JsonlReader, JsonlWriter


def main(args):
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in f:
                document = instance['document']
                labels = instance['labels']
                summary = [document[index] for index in labels]
                out.write({'summary': summary})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The input file with the labeled summaries.')
    argp.add_argument('output_jsonl', help='The output file')
    args = argp.parse_args()
    main(args)
