# Edit the system path so the summarize library can be imported
import sys
sys.path.append('.')

import argparse
import json
import re

from summarize.data.io import JsonlWriter


def main(args):
    with JsonlWriter(args.output_jsonl) as out:
        with open(args.src_tsv, 'r') as f_src:
            with open(args.tgt_tsv, 'r') as f_tgt:
                for src, tgt in zip(f_src, f_tgt):
                    if len(src.strip()) == 0:
                        continue

                    document = [src.strip()]
                    summary = []
                    for match in re.findall(r'<t> (.+?) </t>', tgt):
                        summary.append(match)
                    out.write({'document': document, 'summary': summary})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('src_tsv')
    argp.add_argument('tgt_tsv')
    argp.add_argument('output_jsonl')
    args = argp.parse_args()
    main(args)
