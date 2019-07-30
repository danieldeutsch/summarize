"""
Converts the output of the OpenNMT models to the jsonl format that
is necessary for evaluation. Additionally, the script will remove the
sentence delimiters from the output.
"""
# Edit the system path so the summarize library can be imported
import sys
sys.path.append('.')

import argparse
import json

from summarize.data.io import JsonlWriter


def main(args):
    with JsonlWriter(args.output_jsonl) as out:
        with open(args.input_tsv, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace('<t>', '').replace('</t>', '')
                line = ' '.join(line.split())
                summary = [line]
                out.write({'summary': summary})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_tsv', help='The output from the OpenNMT model')
    argp.add_argument('output_jsonl', help='The converted jsonl file')
    args = argp.parse_args()
    main(args)
