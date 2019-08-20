import argparse

from summarize.data.io import JsonlReader, JsonlWriter
from summarize.metrics.python_rouge import PythonRouge
from summarize.metrics.rouge import \
    R1_RECALL, R1_PRECISION, R1_F1, \
    R2_RECALL, R2_PRECISION, R2_F1, \
    R3_RECALL, R3_PRECISION, R3_F1, \
    R4_RECALL, R4_PRECISION, R4_F1, \
    RL_RECALL, RL_PRECISION, RL_F1
from summarize.models.sds.oracle import get_greedy_oracle_summary


def main(args):
    python_rouge = PythonRouge()

    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in f:
                document = instance['document']
                cloze = instance['cloze']
                oracle, labels = get_greedy_oracle_summary(document, [cloze], args.metric,
                                                           max_sentences=args.max_sentences,
                                                           max_tokens=args.max_tokens,
                                                           max_bytes=args.max_bytes,
                                                           use_porter_stemmer=args.use_stemmer,
                                                           remove_stopwords=args.remove_stopwords,
                                                           python_rouge=python_rouge)
                if args.cloze_only:
                    oracle = ' '.join(oracle)
                    out.write({'cloze': oracle})
                else:
                    instance['labels'] = labels
                    out.write(instance)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The input file to label')
    argp.add_argument('output_jsonl', help='The output file')
    argp.add_argument('metric',
                      choices=[R1_RECALL, R1_PRECISION, R1_F1,
                               R2_RECALL, R2_PRECISION, R2_F1,
                               R3_RECALL, R3_PRECISION, R3_F1,
                               R4_RECALL, R4_PRECISION, R4_F1,
                               RL_RECALL, RL_PRECISION, RL_F1],
                      help='The metric which should be maximized')
    argp.add_argument('--max-sentences', type=int, help='The maximum number of sentences to take')
    argp.add_argument('--max-tokens', type=int, help='The maximum number of tokens to take')
    argp.add_argument('--max-bytes', type=int, help='The maximum number of bytes to take')
    argp.add_argument('--remove-stopwords', action='store_true', dest='remove_stopwords',
                      help='Indicates stopwords should be removed from the summaries.')
    argp.add_argument('--keep-stopwords', action='store_false', dest='remove_stopwords',
                      help='Indicates stopwords should be kept in the summaries.')
    argp.set_defaults(remove_stopwords=False)
    argp.add_argument('--use-stemmer', action='store_true', dest='use_stemmer',
                      help='Indicates that the Porter Stemmer should be used.')
    argp.add_argument('--no-stemmer', action='store_false', dest='use_stemmer',
                      help='Indicates that no stemming should be done.')
    argp.set_defaults(use_stemmer=True)
    argp.add_argument('--cloze-only', action='store_true',
                      help='Indicates the output should contain only the oracle cloze')
    args = argp.parse_args()
    main(args)
