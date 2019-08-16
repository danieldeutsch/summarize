import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ArrayField, ListField, MetadataField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides
from typing import Dict, Iterable, List, Optional

from summarize.data.io import JsonlReader


@DatasetReader.register('sds-extractive')
class ExtractiveDatasetReader(DatasetReader):
    """
    Reads a generic single-document summarization dataset for an extractive
    summarization model. The document should be a sentence tokenized ``List[str]``
    and the ground-truth extractive summary should be a ``List[int]``. An optional
    summary of type ``List[str]`` can also be included.

    Parameters
    ----------
    tokenizer:
        The tokenizer for the document sentences.
    token_indexers:
        The token indexers for the document tokens.
    max_num_sentences:
        The maximum number of allowed sentences per document. If a sentence is
        pruned and it is present in the ground-truth extractive summary, the
        corresponding label is removed.
    max_sentence_length:
        The maximum sentence length after tokenization.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_num_sentences: int = None,
                 max_sentence_length: int = None,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_num_sentences = max_num_sentences
        self.max_sentence_length = max_sentence_length

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        with JsonlReader(file_path) as f:
            for data in f:
                document = data['document']
                labels = data['labels']
                summary = data['summary'] if 'summary' in data else None
                yield self.text_to_instance(document, labels=labels, summary=summary)

    @overrides
    def text_to_instance(self,
                         document: List[str],
                         labels: Optional[List[int]] = None,
                         summary: Optional[List[str]] = None) -> Instance:
        """
        Parameters
        ----------
        document:
            The list of document sentences.
        labels:
            The list of sentence indices that correspond to the ground-truth
            extractive summary.
        summary:
            The list of summary sentences.
        """
        fields = {}

        # If a maximum number of sentences is specified, remove those sentences
        # and any labels which correspond to the indices removed
        if self.max_num_sentences is not None:
            document = document[:self.max_num_sentences]
            if labels is not None:
                labels = list(filter(lambda label: label < self.max_num_sentences, labels))

        tokenized_document = [self.tokenizer.tokenize(sentence) for sentence in document]
        if self.max_sentence_length is not None:
            tokenized_document = [sentence[:self.max_sentence_length] for sentence in tokenized_document]

        text_fields = [TextField(tokens, self.token_indexers) for tokens in tokenized_document]
        document_field = ListField(text_fields)
        fields['document'] = document_field

        if labels is not None:
            label_array = np.zeros(len(document), dtype=np.uint8)
            label_array[labels] = 1
            label_field = ArrayField(label_array)
            fields['labels'] = label_field

        metadata = {}
        metadata['document'] = document
        if summary is not None:
            metadata['summary'] = summary
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
