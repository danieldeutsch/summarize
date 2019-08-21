import argparse

from summarize.data.io import JsonlReader, JsonlWriter


def main(args):
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.source_jsonl) as source:
            with JsonlReader(args.target_jsonl) as target:
                for source_instance, target_instance in zip(source, target):
                    for source_field, target_field in args.field_names:
                        target_instance[target_field] = source_instance[source_field]
                    out.write(target_instance)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('source_jsonl', help='The file with the desired field')
    argp.add_argument('target_jsonl', help='The destination file')
    argp.add_argument('output_jsonl', help='The file with the target data and copied source field')
    argp.add_argument('--field-names', nargs=2, action='append',
                      help='The names of the source and target fields')
    args = argp.parse_args()
    main(args)
