import numpy as np
import unittest
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.dataset_readers.cloze import ExtractiveClozeDatasetReader


class TestExtractiveClozeDatasetReader(unittest.TestCase):
    def test_read_from_file(self):
        tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        reader = ExtractiveClozeDatasetReader(tokenizer=tokenizer, max_num_sentences=5,
                                              max_sentence_length=6, max_context_length=4)
        instances = list(reader.read(f'{FIXTURES_ROOT}/data/cloze.jsonl'))

        instance1 = {
            'document': [
                ['Drew', 'Sheneman', 'has', 'been', 'the', 'editorial'],
                ['J.', ')'],
                ['since', '1998', '.'],
                ['With', 'exceptional', 'artistry', ',', 'his', 'cartoons'],
                ['Sheneman', 'began', 'cartooning', 'in', 'college', 'and']
            ],
            'topics': [['Drew', 'Sheneman']],
            'context': ['American', 'editorial', 'cartoonist', '.'],
            'labels': [1, 0, 1, 0, 1]
        }

        assert len(instances) == 25
        fields = instances[1].fields
        assert len(fields['document'].field_list) == 5
        for sentence, sentence_field in zip(instance1['document'], fields['document'].field_list):
            assert [t.text for t in sentence_field.tokens] == sentence
        assert len(fields['topics'].field_list) == 1
        for topic, topic_field in zip(instance1['topics'], fields['topics'].field_list):
            assert [t.text for t in topic_field.tokens] == topic
        assert [t.text for t in fields['context']] == instance1['context']
        assert np.array_equal(fields['labels'].array, instance1['labels'])
        metadata = fields['metadata']
        assert 'document' in metadata
        assert 'topics' in metadata
        assert 'context' in metadata
        assert 'cloze' in metadata
