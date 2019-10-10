import argparse
import enforce
import json
import logging
import tempfile
from allennlp.common.file_utils import cached_path
from subprocess import Popen, PIPE
from typing import List

from summarize.data.io import JsonlReader

DEFAULT_METEOR_JAR_PATH = 'external/meteor/meteor-1.5/meteor-1.5.jar'


def _load_summaries(file_path: str) -> List[str]:
    summaries = []
    with JsonlReader(cached_path(file_path)) as f:
        for data in f:
            summaries.append(' '.join(data['summary']))
    return summaries


def _save_summaries(summaries: List[str], file_path: str) -> None:
    with open(file_path, 'w') as out:
        for summary in summaries:
            out.write(summary + '\n')


def _parse_meteor_stdout(stdout: str) -> float:
    lines = stdout.splitlines()
    last_line = lines[-1]
    if not last_line.startswith('Final score'):
        raise Exception(f'Unexpected stdout format: {stdout}')

    columns = last_line.split()
    score = float(columns[-1]) * 100
    return score


@enforce.runtime_validation
def run_meteor(gold_summaries: List[str],
               model_summaries: List[str],
               meteor_jar_path: str = DEFAULT_METEOR_JAR_PATH) -> float:
    if len(gold_summaries) != len(model_summaries):
        raise Exception(f'Unequal gold and model summaries. '
                        f'Found {len(gold_summaries)} and {len(model_summaries)}')

    with tempfile.NamedTemporaryFile() as gold_file_path:
        with tempfile.NamedTemporaryFile() as model_file_path:
            _save_summaries(gold_summaries, gold_file_path.name)
            _save_summaries(model_summaries, model_file_path.name)

            # The test output comes before the model output
            command = [
                'java', '-jar', meteor_jar_path,
                model_file_path.name, gold_file_path.name,
                '-l', 'en',
                '-norm'
            ]
            command_string = ' '.join(command)
            logging.info(f'Running Meteor with command: "{command_string}"')

            process = Popen(command, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            if stderr:
                raise Exception(f'Meteor failed with stderr: {stderr.decode()}')

            score = _parse_meteor_stdout(stdout.decode())
            return score


def main(args):
    if args.silent and not args.output_file:
        raise Exception(f'No output will be written with --silent and no output file')

    gold_summaries = _load_summaries(args.gold_summaries)
    model_summaries = _load_summaries(args.model_summaries)

    score = run_meteor(gold_summaries, model_summaries)
    metrics = {
        'meteor': score
    }

    if not args.silent:
        print(json.dumps(metrics, indent=4))
    if args.output_file:
        with open(args.output_file, 'w') as out:
            out.write(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('gold_summaries', help='The path to the gold summary jsonl file.')
    argp.add_argument('model_summaries', help='The path to the model summary jsonl file.')
    argp.add_argument('--output-file',
                      help='The path to the file where the json output should be written.')
    argp.add_argument('--silent', action='store_true',
                      help='Indicates nothing should be written to stdout.')
    args = argp.parse_args()
    main(args)
