import unittest
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.dataset_readers.cloze import AbstractiveClozeDatasetReader
from summarize.data.paragraph_tokenizers import ParagraphWordTokenizer


class TestAbstractiveClozeDatasetReader(unittest.TestCase):
    def test_read_from_file(self):
        word_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        paragraph_tokenizer = ParagraphWordTokenizer(word_splitter=JustSpacesWordSplitter())
        reader = AbstractiveClozeDatasetReader(document_tokenizer=paragraph_tokenizer,
                                               topic_tokenizer=word_tokenizer,
                                               max_document_length=10,
                                               max_context_length=7,
                                               max_cloze_length=5)
        instances = list(reader.read(f'{FIXTURES_ROOT}/data/cloze.jsonl'))

        instance0 = {
            'document': ['NEW', 'YORK', ',', 'Jan.', '8', ',', '2016', '/PRNewswire/', '--', 'Businessman'],
            'topics': [['Ken', 'Fields'], ['Politics']],
            'context': ['%', 'Renewable', 'Energy', 'in', '20', 'Years', '.'],
            'cloze': ['Picking', 'as', 'his', 'campaign', 'slogan']
        }

        assert len(instances) == 25
        fields = instances[0].fields
        assert [t.text for t in fields['document'].tokens] == instance0['document']
        assert len(fields['topics'].field_list) == len(instance0['topics'])
        for topic_field, topic in zip(fields['topics'].field_list, instance0['topics']):
            assert [t.text for t in topic_field.tokens] == topic
        assert [t.text for t in fields['context'].tokens] == instance0['context']
        assert [t.text for t in fields['cloze'].tokens] == instance0['cloze']
        metadata = fields['metadata']
        assert 'document' in metadata
        assert 'topics' in metadata
        assert 'context' in metadata
        assert 'cloze' in metadata
