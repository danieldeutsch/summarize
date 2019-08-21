import numpy as np
import unittest
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.dataset_readers.cloze import PointerGeneratorClozeDatasetReader
from summarize.data.paragraph_tokenizers import ParagraphWordTokenizer


class TestPointerGeneratorDatasetReader(unittest.TestCase):
    def test_read_from_file(self):
        word_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        paragraph_tokenizer = ParagraphWordTokenizer(word_splitter=JustSpacesWordSplitter())
        document_token_idexers = {'tokens': SingleIdTokenIndexer('document')}
        cloze_token_indexers = {'tokens': SingleIdTokenIndexer('cloze')}
        reader = PointerGeneratorClozeDatasetReader(document_tokenizer=paragraph_tokenizer,
                                                    topic_tokenizer=word_tokenizer,
                                                    document_token_indexers=document_token_idexers,
                                                    cloze_token_indexers=cloze_token_indexers,
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

        # Use another instance to check the document indices for each cloze
        reader = PointerGeneratorClozeDatasetReader(document_tokenizer=paragraph_tokenizer,
                                                    topic_tokenizer=word_tokenizer,
                                                    document_token_indexers=document_token_idexers,
                                                    cloze_token_indexers=cloze_token_indexers)
        instances = list(reader.read(f'{FIXTURES_ROOT}/data/cloze.jsonl'))
        fields = instances[1].fields

        assert fields['document_token_first_indices'].array.shape == (205,)
        assert fields['document_token_first_indices'].array[0] == 0
        assert fields['document_token_first_indices'].array[1] == 1
        assert fields['document_token_first_indices'].array[2] == 2
        assert fields['document_token_first_indices'].array[3] == 3
        # Indices of "Sherman"
        assert fields['document_token_first_indices'].array[40] == 1
        assert fields['document_token_first_indices'].array[114] == 1
        assert fields['document_token_first_indices'].array[170] == 1

        assert len(fields['context_token_document_indices'].field_list) == 12
        assert np.array_equal(fields['context_token_document_indices'].field_list[0].array, [0])
        assert np.array_equal(fields['context_token_document_indices'].field_list[1].array, [1, 40, 114, 170])
        assert np.array_equal(fields['context_token_document_indices'].field_list[2].array, [12, 58])
        assert np.array_equal(fields['context_token_document_indices'].field_list[11].array, [20, 39, 90, 113, 169, 194, 204])

        assert len(fields['context_token_document_indices_mask'].field_list) == 12
        assert np.array_equal(fields['context_token_document_indices_mask'].field_list[0].array, [1])
        assert np.array_equal(fields['context_token_document_indices_mask'].field_list[1].array, [1, 1, 1, 1])
        assert np.array_equal(fields['context_token_document_indices_mask'].field_list[2].array, [1, 1])
        assert np.array_equal(fields['context_token_document_indices_mask'].field_list[11].array, [1, 1, 1, 1, 1, 1, 1])

        assert len(fields['cloze_token_document_indices'].field_list) == 28
        assert np.array_equal(fields['cloze_token_document_indices'].field_list[0].array, [91])
        assert np.array_equal(fields['cloze_token_document_indices'].field_list[1].array, [92, 185])
        assert np.array_equal(fields['cloze_token_document_indices'].field_list[2].array, [14, 24, 60, 74, 104, 133, 147, 189])
        assert np.array_equal(fields['cloze_token_document_indices'].field_list[27].array, [20, 39, 90, 113, 169, 194, 204])

        assert len(fields['cloze_token_document_indices_mask'].field_list) == 28
        assert np.array_equal(fields['cloze_token_document_indices_mask'].field_list[0].array, [1])
        assert np.array_equal(fields['cloze_token_document_indices_mask'].field_list[1].array, [1, 1])
        assert np.array_equal(fields['cloze_token_document_indices_mask'].field_list[2].array, [1, 1, 1, 1, 1, 1, 1, 1])
        assert np.array_equal(fields['cloze_token_document_indices_mask'].field_list[27].array, [1, 1, 1, 1, 1, 1, 1])
