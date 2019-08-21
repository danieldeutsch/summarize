"""
A python-wrapper around the official perl ROUGE script.
"""
import argparse
import enforce
import json
import logging
import os
from allennlp.common.file_utils import cached_path
from subprocess import Popen, PIPE
from typing import Any, Dict, List, Optional, Tuple, Union

from summarize.common import TemporaryDirectory
from summarize.data.io import JsonlReader


R1_RECALL, R1_PRECISION, R1_F1 = 'R1-R', 'R1-P', 'R1-F1'
R2_RECALL, R2_PRECISION, R2_F1 = 'R2-R', 'R2-P', 'R2-F1'
R3_RECALL, R3_PRECISION, R3_F1 = 'R3-R', 'R3-P', 'R3-F1'
R4_RECALL, R4_PRECISION, R4_F1 = 'R4-R', 'R4-P', 'R4-F1'
RL_RECALL, RL_PRECISION, RL_F1 = 'RL-R', 'RL-P', 'RL-F1'

logger = logging.getLogger(__name__)


def _load_summaries(file_path: str,
                    field_name: str = 'summary',
                    add_wrapping_list: bool = False) -> Union[List[List[str]], List[List[List[str]]]]:
    summaries = []
    with JsonlReader(cached_path(file_path)) as f:
        for data in f:
            summary = data[field_name]
            if add_wrapping_list:
                summary = [summary]
            summaries.append(summary)
    return summaries


@enforce.runtime_validation
def _save_summary(summary: List[str], file_path: str) -> None:
    with open(file_path, 'w') as out:
        for sentence in summary:
            out.write(sentence + '\n')


def _save_config_file(file_path: str,
                      gold_filenames_list: List[List[str]],
                      model_filenames: List[str]) -> None:
    output_dir = os.path.dirname(file_path)
    with open(file_path, 'w') as out:
        out.write(f'<ROUGE_EVAL version="1.0">\n')
        for i, (gold_filenames, model_filename) in enumerate(zip(gold_filenames_list, model_filenames)):
            out.write(f'<EVAL ID="{i + 1}">\n')
            out.write(f'<INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>\n')
            out.write(f'<PEER-ROOT>{output_dir}</PEER-ROOT>\n')
            out.write(f'<MODEL-ROOT>{output_dir}</MODEL-ROOT>\n')
            out.write(f'<PEERS>\n')
            out.write(f'<P ID="1">{model_filename}</P>\n')
            out.write(f'</PEERS>\n')
            out.write(f'<MODELS>\n')
            for j, gold_filename in enumerate(gold_filenames):
                symbol = chr(j + 65)
                out.write(f'<M ID="{symbol}">{gold_filename}</M>\n')
            out.write(f'</MODELS>\n')
            out.write(f'</EVAL>\n')
        out.write(f'</ROUGE_EVAL>\n')


def _parse_rouge_stdout_line(line: str) -> Tuple[str, float, float, float]:
    columns = line.split()
    assert len(columns) == 8
    metric = columns[1][-1]
    value = float(columns[3])
    lower_ci = float(columns[5])
    upper_ci = float(columns[7][:-1])
    return metric, value, lower_ci, upper_ci


def _parse_rouge_stdout(stdout: str) -> Dict[str, float]:
    lines = stdout.splitlines()
    metrics = {}
    for i in range(0, len(lines), 4):
        assert lines[i] == '---------------------------------------------'
        recall_line = lines[i + 1]
        precision_line = lines[i + 2]
        f1_line = lines[i + 3]
        metric, value, lower_ci, upper_ci = _parse_rouge_stdout_line(recall_line)
        metrics[f'R{metric}-R'] = value * 100
        metric, value, lower_ci, upper_ci = _parse_rouge_stdout_line(precision_line)
        metrics[f'R{metric}-P'] = value * 100
        metric, value, lower_ci, upper_ci = _parse_rouge_stdout_line(f1_line)
        metrics[f'R{metric}-F1'] = value * 100
    return metrics


def has_multiple_references(summaries: Union[List[List[str]], List[List[List[str]]]]) -> bool:
    """
    Checks to see if ``summaries`` has multiple references or not by examining
    the type of the summaries. Each individual summary is represented as ``List[str]``.
    If the type is ``List[List[str]]``, there is one summary per instance. If the
    type is ``List[List[List[str]]]``, there are multiple summaries per instance.
    """
    def _is_summary(maybe_summary: Union[List[str], Any]) -> bool:
        return isinstance(maybe_summary, list) and all(isinstance(item, str) for item in maybe_summary)

    if isinstance(summaries, list):
        if all(_is_summary(maybe_summary) for maybe_summary in summaries):
            return False
        for instance_summaries in summaries:
            if not all(_is_summary(maybe_summary) for maybe_summary in instance_summaries):
                raise Exception('``summaries`` is neither ``List[List[str]]`` nor ``List[List[List[str]]]``.')
        return True


def run_rouge(gold_summaries: Union[List[List[str]], List[List[List[str]]]],
              model_summaries: List[List[str]],
              rouge_script_location: str = 'external/ROUGE-1.5.5/ROUGE-1.5.5.pl',
              rouge_eval_home: str = 'external/ROUGE-1.5.5/data',
              max_ngram: int = 4,
              use_porter_stemmer: bool = True,
              remove_stopwords: bool = False,
              max_bytes: Optional[int] = None,
              max_words: Optional[int] = None,
              compute_rouge_l: bool = False) -> Dict[str, float]:
    """
    Runs the official perl ROUGE evaluation. Each individual summary should be
    represented as a ``List[str]``. The default parameters of this function are
    set to match the default parameters of the perl script.

    Parameters
    ----------
    gold_summaries: ``Union[List[List[str]], List[List[List[str]]]]``, required
        The ground-truth summaries. If there is only one ground-truth summary per
        instance, the type should be ``List[List[str]]``. If there are multiple, then
        ``List[List[List[str]]]``.
    model_summaries: ``List[List[str]]``, required
        The model summaries to evaluate.
    rouge_script_location: ``str``, optional (default = "external/ROUGE-1.5.5/ROUGE-1.5.5.pl")
        The path to the official ROUGE perl script.
    rouge_eval_home: ``str``, optional (default = "external/ROUGE-1.5.5/data")
        The path to the ROUGE data. The name for this parameter comes from the official
        ROUGE script.
    max_ngram: ``int``, optional (default = ``4``)
        The maximum n-gram order of ROUGE to compute.
    use_porter_stemmer: ``bool``, optional (default = ``True``)
        Indicates if the Porter Stemmer should be used to preprocess the summaries.
    remove_stopwords: ``bool``, optional (default = ``False``)
        Indicates if stopwords should be removed from the summaries. The stopword
        list comes from "smart_common_words.txt" in the ROUGE data directory.
    max_bytes: ``int``, optional (default = ``None``)
        Limits the length of the summaries based on the number of bytes. If ``None``,
        no truncation is performed. This option can not be used together with ``max_words``.
    max_words: ``int``, optional (default = ``None``)
        Limits the length of the summaries based on the number of words separated by whitespace.
        If ``None``, no truncation is performed. This option can not be used together with ``max_bytes``.
    compute_rouge_l: ``bool``, optional (default = ``False``)
        Indicates whether or not the ROUGE-L scores should be computed.

    Returns
    -------
    A dictionary that maps from the metric name to the value.
    """
    if len(gold_summaries) != len(model_summaries):
        raise Exception(f'Unequal gold and model summaries. '
                        f'Found {len(gold_summaries)} and {len(model_summaries)}')
    if max_bytes is not None and max_words is not None:
        raise Exception(f'The maximum number of bytes and words cannot both be set.')

    multiple_references = has_multiple_references(gold_summaries)

    with TemporaryDirectory() as temp_dir:
        model_filenames = []
        gold_filenames_list = []

        for i, (gold_summary, model_summary) in enumerate(zip(gold_summaries, model_summaries)):
            model_filename = f'model.{i}.txt'
            _save_summary(model_summary, os.path.join(temp_dir, model_filename))
            model_filenames.append(model_filename)

            if not multiple_references:
                gold_summary = [gold_summary]
            gold_filenames = []
            for j, summary in enumerate(gold_summary):
                symbol = chr(j + 65)
                gold_filename = f'gold.{symbol}.{i}.txt'
                _save_summary(summary, os.path.join(temp_dir, f'gold.{symbol}.{i}.txt'))
                gold_filenames.append(gold_filename)
            gold_filenames_list.append(gold_filenames)

        config_file_path = os.path.join(temp_dir, 'config.xml')
        _save_config_file(config_file_path, gold_filenames_list, model_filenames)

        command = [
            rouge_script_location,
            '-e', rouge_eval_home,
            '-n', str(max_ngram),
            '-a',
            '-c', '95',
            '-r', '1000',
            '-f', 'A',
            '-p', '0.5',
            '-t', '0'
        ]
        if use_porter_stemmer:
            command += ['-m']
        if remove_stopwords:
            command += ['-s']
        if max_bytes is not None:
            command += ['-b', str(max_bytes)]
        if max_words is not None:
            command += ['-l', str(max_words)]
        if not compute_rouge_l:
            command += ['-x']
        command += [config_file_path]

        command_string = ' '.join(command)
        logging.info(f'Running Rouge with command: "{command_string}"')

        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        if stderr:
            raise Exception(f'Rouge failed with stderr: {stderr.decode()}')

        results = _parse_rouge_stdout(stdout.decode())
        return results


def main(args):
    if args.silent and not args.output_file:
        raise Exception(f'No output will be written with --silent and no output file')

    gold_summaries = _load_summaries(args.gold_summaries, args.gold_summary_field_name, args.add_gold_wrapping_list)
    model_summaries = _load_summaries(args.model_summaries, args.model_summary_field_name, args.add_model_wrapping_list)

    metrics = run_rouge(gold_summaries,
                        model_summaries,
                        max_ngram=args.max_ngram,
                        use_porter_stemmer=args.use_stemmer,
                        remove_stopwords=args.remove_stopwords,
                        max_bytes=args.max_bytes,
                        max_words=args.max_words,
                        compute_rouge_l=args.compute_rouge_l)

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
    argp.add_argument('--gold-summary-field-name', default='summary',
                      help='The field name in the gold json for the summaries (e.g., "summary" for '
                           'single reference or "summaries" for multi-reference).')
    argp.add_argument('--model-summary-field-name', default='summary',
                      help='The field name in the model json for the summaries (e.g., "summary" for '
                           'single reference or "summaries" for multi-reference).')
    argp.add_argument('--silent', action='store_true',
                      help='Indicates nothing should be written to stdout.')
    argp.add_argument('--max-ngram', type=int, default=4,
                      help='The maximum order of ngram to use.')
    argp.add_argument('--max-bytes', type=int, default=None,
                      help='The maximum number of bytes to use from the summary.')
    argp.add_argument('--max-words', type=int, default=None,
                      help='The maximum number of words to use from the summary.')
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
    argp.add_argument('--compute-rouge-l', action='store_true',
                      help='Indicates that Rouge-L should be computed.')
    argp.set_defaults(compute_rouge_l=False)
    argp.add_argument('--add-gold-wrapping-list', action='store_true',
                      help='Indicates that the loaded gold summaries should be enclosed in a list, '
                           'useful for when the summary is a raw string')
    argp.add_argument('--add-model-wrapping-list', action='store_true',
                      help='Indicates that the loaded model summaries should be enclosed in a list, '
                           'useful for when the summary is a raw string')
    args = argp.parse_args()
    main(args)
