import unittest
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.dataset_readers.sds import PointerGeneratorDatasetReader
from summarize.data.paragraph_tokenizers import ParagraphWordTokenizer


class TestPointerGeneratorDatasetReader(unittest.TestCase):
    def test_read_from_file(self):
        tokenizer = ParagraphWordTokenizer(word_splitter=JustSpacesWordSplitter())
        document_token_idexers = {'tokens': SingleIdTokenIndexer('document')}
        summary_token_indexers = {'tokens': SingleIdTokenIndexer('summary')}
        reader = PointerGeneratorDatasetReader(document_tokenizer=tokenizer,
                                               document_token_indexers=document_token_idexers,
                                               summary_token_indexers=summary_token_indexers,
                                               max_document_length=10, max_summary_length=5)
        instances = list(reader.read(f'{FIXTURES_ROOT}/data/sds.jsonl'))

        instance0 = {
            'document': ['Editor', '\'s', 'note', ':', 'In', 'our', 'Behind', 'the', 'Scenes', 'series'],
            'summary': ['Mentally', 'ill', 'inmates', 'in', 'Miami']
        }
        instance1 = {
            'document': ['LONDON', ',', 'England', '(', 'Reuters', ')', '--', 'Harry', 'Potter', 'star'],
            'summary': ['Harry', 'Potter', 'star', 'Daniel', 'Radcliffe']
        }

        assert len(instances) == 25
        fields = instances[0].fields
        assert [t.text for t in fields['document'].tokens] == instance0['document']
        assert [t.text for t in fields['summary'].tokens] == instance0['summary']
        assert len(fields['document_in_summary_namespace']._source_tokens) == 10
        metadata = fields['metadata']
        assert 'document' in metadata
        assert len(metadata['document']) == 20
        assert 'summary' in metadata
        assert len(metadata['summary']) == 4
        assert (fields['document_token_first_indices'].array == list(range(10))).all()

        # Use another instance to check the document indices for each summary token
        fields = instances[1].fields
        assert [t.text for t in fields['document'].tokens] == instance1['document']
        assert [t.text for t in fields['summary'].tokens] == instance1['summary']
        summary_token_document_indices = fields['summary_token_document_indices']
        assert len(summary_token_document_indices.field_list) == 5
        assert summary_token_document_indices.field_list[0].array == [7]
        assert summary_token_document_indices.field_list[1].array == [8]
        assert summary_token_document_indices.field_list[2].array == [9]
        assert summary_token_document_indices.field_list[3].array.size == 0
        assert summary_token_document_indices.field_list[4].array.size == 0
        summary_token_document_indices_mask = fields['summary_token_document_indices_mask']
        assert len(summary_token_document_indices_mask.field_list) == 5
        assert summary_token_document_indices_mask.field_list[0].array == [1]
        assert summary_token_document_indices_mask.field_list[1].array == [1]
        assert summary_token_document_indices_mask.field_list[2].array == [1]
        assert summary_token_document_indices_mask.field_list[3].array.size == 0
        assert summary_token_document_indices_mask.field_list[4].array.size == 0
