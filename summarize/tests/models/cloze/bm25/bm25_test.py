import tempfile
import unittest
from collections import namedtuple

from summarize.data.io import JsonlReader
from summarize.common.testing import FIXTURES_ROOT
from summarize.models.cloze.bm25 import calculate_df, bm25


class TestBM25(unittest.TestCase):
    def test_bm25_runs(self):
        with tempfile.NamedTemporaryFile(suffix='.jsonl') as df_file:
            with tempfile.NamedTemporaryFile(suffix='.jsonl') as bm25_file:
                Args = namedtuple('Args', ['input_jsonl', 'output_jsonl'])
                args = Args(f'{FIXTURES_ROOT}/data/cloze.jsonl', df_file.name)
                calculate_df.main(args)

                Args = namedtuple('Args', ['input_jsonl', 'df_jsonl', 'output_jsonl',
                                           'k', 'b', 'max_words', 'max_sentences', 'flatten'])
                args = Args(f'{FIXTURES_ROOT}/data/cloze.jsonl', df_file.name, bm25_file.name,
                            1.2, 0.75, None, 1, True)
                bm25.main(args)

                instances = JsonlReader(bm25_file.name).read()
                assert len(instances) == 25
                for instance in instances:
                    assert 'cloze' in instance
                    assert isinstance(instance['cloze'], str)
