import tempfile
import unittest
from collections import namedtuple

from summarize.data.io import JsonlReader
from summarize.models.cloze import lead
from summarize.common.testing import FIXTURES_ROOT


class TestClozeLead(unittest.TestCase):
    def test_cloze_lead(self):
        with tempfile.NamedTemporaryFile(suffix='.jsonl') as output_file:
            Args = namedtuple('Args', ['input_jsonl', 'output_jsonl', 'max_sentences',
                                       'max_tokens', 'max_bytes', 'field_name', 'keep_sentences'])
            args = Args(f'{FIXTURES_ROOT}/data/cloze.jsonl', output_file.name,
                        1, None, None, 'cloze', True)
            lead.main(args)

            instances = JsonlReader(output_file.name).read()
            assert len(instances) == 25
            assert all('cloze' in instance for instance in instances)
            assert all(isinstance(instance['cloze'], list) for instance in instances)

            args = Args(f'{FIXTURES_ROOT}/data/cloze.jsonl', output_file.name,
                        1, None, None, 'cloze', False)
            lead.main(args)

            instances = JsonlReader(output_file.name).read()
            assert len(instances) == 25
            assert all('cloze' in instance for instance in instances)
            assert all(isinstance(instance['cloze'], str) for instance in instances)
