import argparse

from summarize.data.io import JsonlReader, JsonlWriter
from summarize.models.sds.lead import get_lead_summary


def main(args):
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in f:
                document = instance['document']
                cloze = get_lead_summary(document,
                                         max_sentences=args.max_sentences,
                                         max_tokens=args.max_tokens,
                                         max_bytes=args.max_bytes)
                if not args.keep_sentences:
                    cloze = ' '.join(cloze)
                out.write({args.field_name: cloze})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The input documents')
    argp.add_argument('output_jsonl', help='The output file')
    argp.add_argument('--max-sentences', type=int, help='The number of sentences to take')
    argp.add_argument('--max-tokens', type=int, help='The number of tokens to take')
    argp.add_argument('--max-bytes', type=int, help='The number of bytes to take')
    argp.add_argument('--field-name', default='cloze', help='The name of the output field')
    argp.add_argument('--keep-sentences', action='store_true', help='Indicates if the output field should be left as sentences or flattened')
    args = argp.parse_args()
    main(args)
