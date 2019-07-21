import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import ArrayField, ListField, MetadataField, NamespaceSwappingField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from collections import defaultdict
from overrides import overrides
from typing import Dict, Iterable, List, Optional, Tuple

from summarize.data.io import JsonlReader
from summarize.data.paragraph_tokenizers import ParagraphTokenizer, ParagraphWordTokenizer


@DatasetReader.register('sds-pointer-generator')
class PointerGeneratorDatasetReader(DatasetReader):
    """
    Reads a generic single-document summarization dataset for a the Pointer-Generator
    summarization model. In addition to the fields provided by the ``AbstractiveDatasetReader``,
    this class also returns the first index of each document token in the document,
    the document tokens indexed in the summary vocabulary namespace, and the
    indices in the document for each of the summary tokens.

    Both the document and the summary is expected to be a
    list of sentences of type ``List[str]`` in "document" and "summary" field names.

    Parameters
    ----------
    document_token_indexers: ``Dict[str, TokenIndexer]``, optional (default = ``{'tokens': SingleIdTokenIndexer()}``).
        The token indexers used for the document tokens.
    summary_token_indexers: ``Dict[str, TokenIndexer]``, optional.
        The token indexers used for the summary tokens. If not provided, the default value
        is set to be the same object as ``document_token_indexers``.
    document_tokenizer: ``ParagraphTokenizer``, optional (default = ``ParagraphWordTokenizer``).
        The tokenizer for the document text.
    summary_tokenizer: ``ParagraphTokenizer``, optional.
        The tokenizer for the summary text. If not provided, the default value is set
        to be ``document_tokenizer``.
    max_document_length: ``int``, optional (default = ``None``).
        The maximum number of document tokens to use. If ``None``, no truncation is performed. The
        truncation runs after the tokenization step, so this length number includes any ``start_tokens``,
        ``end_tokens``, etc. It does not ensure that the ``end_tokens`` will still be at the end
        of the sequence.
    max_summary_length: ``int``, optional (default = ``None``).
        The maximum number of summary tokens to use. See ``max_document_length`` for more details.
    """
    def __init__(self,
                 document_token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 summary_token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 document_tokenizer: Optional[ParagraphTokenizer] = None,
                 summary_tokenizer: Optional[ParagraphTokenizer] = None,
                 max_document_length: Optional[int] = None,
                 max_summary_length: Optional[int] = None,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.document_token_indexers = document_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.summary_token_indexers = summary_token_indexers or self.document_token_indexers
        self.document_tokenizer = document_tokenizer or ParagraphWordTokenizer()
        self.summary_tokenizer = summary_tokenizer or self.document_tokenizer
        self.max_document_length = max_document_length
        self.max_summary_length = max_summary_length
        self.summary_namespace = self.summary_token_indexers['tokens'].namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        with JsonlReader(file_path) as f:
            for data in f:
                document = data['document']
                summary = data['summary']
                yield self.text_to_instance(document, summary=summary)

    def _get_token_to_index_map(self, document: List[Token]) -> Dict[str, List[int]]:
        # Build an index from document token to the index
        token_to_document_indices = defaultdict(list)
        for i, token in enumerate(document):
            token_to_document_indices[str(token)].append(i)
        return token_to_document_indices

    def _get_document_token_first_indices_field(self,
                                                document: List[Token],
                                                token_to_indices: Dict[str, List[int]]) -> Dict[str, int]:
        first_indices = []
        for token in document:
            first_index = token_to_indices[str(token)][0]
            first_indices.append(first_index)
        return ArrayField(np.array(first_indices))

    def _get_token_mapping_field(self,
                                 token_to_document_indices: Dict[str, List[int]],
                                 summary: List[Token]) -> Tuple[ListField, ListField]:
        # For every token in the summary, retrieve the list of document positions
        # where that token is. The mask will indicate which are true indices
        # and which are invalid
        summary_token_document_indices = []
        mask = []
        for token in summary:
            indices = token_to_document_indices[str(token)]
            summary_token_document_indices.append(ArrayField(np.array(indices)))
            mask.append(ArrayField(np.ones(len(indices))))

        # Convert these into fields
        summary_token_document_indices_field = ListField(summary_token_document_indices)
        mask_field = ListField(mask)
        return summary_token_document_indices_field, mask_field

    @overrides
    def text_to_instance(self, document: List[str], summary: Optional[List[str]]) -> Instance:
        """
        Parameters
        ----------
        document: ``List[str]``, required.
            The list of document sentences.
        summary: ``List[str]``, optional.
            The list of summary sentences.
        """
        fields = {}

        # Setup the document field
        tokenized_document = self.document_tokenizer.tokenize(document)
        if self.max_document_length is not None:
            tokenized_document = tokenized_document[:self.max_document_length]
        document_field = TextField(tokenized_document, self.document_token_indexers)
        fields['document'] = document_field

        # Get the document token indices but in the summary namespace
        fields['document_in_summary_namespace'] = NamespaceSwappingField(tokenized_document, self.summary_namespace)

        # Build a map from token to all of the indices that token appears
        document_token_to_indices = self._get_token_to_index_map(tokenized_document)

        # Get a field that, for every document token, has the first index within
        # the document that token appears
        fields['document_token_first_indices'] = \
            self._get_document_token_first_indices_field(tokenized_document, document_token_to_indices)

        # Setup the summary field, if it exists
        if summary is not None:
            tokenized_summary = self.summary_tokenizer.tokenize(summary)
            if self.max_summary_length is not None:
                tokenized_summary = tokenized_summary[:self.max_summary_length]
            summary_field = TextField(tokenized_summary, self.summary_token_indexers)
            fields['summary'] = summary_field

            summary_token_document_indices_field, mask_field = \
                self._get_token_mapping_field(document_token_to_indices, tokenized_summary)
            fields['summary_token_document_indices'] = summary_token_document_indices_field
            fields['summary_token_document_indices_mask'] = mask_field

        # Pass the original data through as metadata
        metadata = {}
        metadata['document'] = document
        metadata['document_tokens'] = [str(token) for token in tokenized_document]
        if summary is not None:
            metadata['summary'] = summary
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
