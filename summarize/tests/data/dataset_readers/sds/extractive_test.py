import numpy as np
import unittest
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.dataset_readers.sds import ExtractiveDatasetReader


class TestExtractiveDatasetReader(unittest.TestCase):
    def test_read_from_file(self):
        tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        reader = ExtractiveDatasetReader(tokenizer=tokenizer, max_num_sentences=5, max_sentence_length=6)
        instances = list(reader.read(f'{FIXTURES_ROOT}/data/sds.jsonl'))

        instance0 = {
            'document': [
                ['Editor', '\'s', 'note', ':', 'In', 'our'],
                ['An', 'inmate', 'housed', 'on', 'the', '``'],
                ['MIAMI', ',', 'Florida', '(', 'CNN', ')'],
                ['Most', 'often', ',', 'they', 'face', 'drug'],
                ['So', ',', 'they', 'end', 'up', 'on']
            ]
        }

        assert len(instances) == 25
        fields = instances[0].fields
        assert len(fields['document'].field_list) == 5
        for sentence, sentence_field in zip(instance0['document'], fields['document'].field_list):
            assert [t.text for t in sentence_field.tokens] == sentence
        assert np.array_equal(fields['labels'].array, [0, 0, 1, 1, 0])
        metadata = fields['metadata']
        assert 'document' in metadata
        assert len(metadata['document']) == 5
        assert 'summary' in metadata
        assert len(metadata['summary']) == 4
