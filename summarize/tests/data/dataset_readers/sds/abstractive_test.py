import unittest
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.dataset_readers.sds import AbstractiveDatasetReader
from summarize.data.paragraph_tokenizers import ParagraphWordTokenizer


class TestAbstractiveDatasetReader(unittest.TestCase):
    def test_read_from_file(self):
        tokenizer = ParagraphWordTokenizer(word_splitter=JustSpacesWordSplitter())
        reader = AbstractiveDatasetReader(document_tokenizer=tokenizer, max_document_length=10, max_summary_length=5)
        instances = list(reader.read(f'{FIXTURES_ROOT}/data/sds.jsonl'))

        instance0 = {
            'document': ['Editor', '\'s', 'note', ':', 'In', 'our', 'Behind', 'the', 'Scenes', 'series'],
            'summary': ['Mentally', 'ill', 'inmates', 'in', 'Miami']
        }

        assert len(instances) == 25
        fields = instances[0].fields
        assert [t.text for t in fields['document'].tokens] == instance0['document']
        assert [t.text for t in fields['summary'].tokens] == instance0['summary']
        metadata = fields['metadata']
        assert 'document' in metadata
        assert len(metadata['document']) == 20
        assert 'summary' in metadata
        assert len(metadata['summary']) == 4
