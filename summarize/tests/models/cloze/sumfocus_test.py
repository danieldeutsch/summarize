import tempfile
import unittest
from collections import namedtuple

from summarize.data.io import JsonlReader
from summarize.common.testing import FIXTURES_ROOT
from summarize.models.cloze import sumfocus


class TestSumFocus(unittest.TestCase):
    def test_sumfocus_runs(self):
        with tempfile.NamedTemporaryFile(suffix='.jsonl') as output_file:
            Args = namedtuple('Args', ['input_jsonl', 'output_jsonl', 'beta',
                                       'topic_lambda', 'context_lambda',
                                       'max_words', 'max_sentences'])
            args = Args(f'{FIXTURES_ROOT}/data/cloze.jsonl', output_file.name,
                        0.5, 0.2, 0.3, 200, None)
            sumfocus.main(args)

            instances = JsonlReader(output_file.name).read()
            assert len(instances) == 25
            for instance in instances:
                assert 'cloze' in instance
                assert isinstance(instance['cloze'], str)
