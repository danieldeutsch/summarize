import tempfile
import unittest
from collections import namedtuple

from summarize.data.io import JsonlReader
from summarize.common.testing import FIXTURES_ROOT
from summarize.models.cloze.bm25 import calculate_df


class TestCalculateDF(unittest.TestCase):
    def test_calculate_df_runs(self):
        with tempfile.NamedTemporaryFile(suffix='.jsonl') as df_file:
            Args = namedtuple('Args', ['input_jsonl', 'output_jsonl'])
            args = Args(f'{FIXTURES_ROOT}/data/cloze.jsonl', df_file.name)
            calculate_df.main(args)

            lines = JsonlReader(df_file.name).read()
            assert len(lines) > 0
            metadata = lines[0]
            assert 'num_documents' in metadata
            assert 'average_document_length' in metadata
            for count in lines[1:]:
                assert 'token' in count
                assert 'df' in count
