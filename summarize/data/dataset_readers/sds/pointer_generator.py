from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, NamespaceSwappingField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from overrides import overrides
from typing import Dict, Iterable, List, Optional

from summarize.data.dataset_readers.util import get_first_indices_field, get_token_mapping_field, get_token_to_index_map
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

    @overrides
    def text_to_instance(self, document: List[str], summary: Optional[List[str]] = None) -> Instance:
        """
        Parameters
        ----------
        document: ``List[str]``, required.
            The list of document sentences.
        summary: ``List[str]``, optional.
            The list of summary sentences.
        """
        fields = {}

        # There is some weirdness that can happen if the document tokens are lowercased
        # but the summary tokens are not (or vice versa). We will deal with that when
        # it's necessary. For now, we don't allow it.
        assert self.document_token_indexers['tokens'].lowercase_tokens == self.summary_token_indexers['tokens'].lowercase_tokens
        if self.document_token_indexers['tokens'].lowercase_tokens:
            document = [sentence.lower() for sentence in document]
            if summary is not None:
                summary = [sentence.lower() for sentence in summary]

        # Setup the document field
        tokenized_document = self.document_tokenizer.tokenize(document)
        if self.max_document_length is not None:
            tokenized_document = tokenized_document[:self.max_document_length]
        document_field = TextField(tokenized_document, self.document_token_indexers)
        fields['document'] = document_field

        # Get the document token indices but in the summary namespace
        fields['document_in_summary_namespace'] = NamespaceSwappingField(tokenized_document, self.summary_namespace)

        # Build a map from token to all of the indices that token appears
        document_token_to_indices = get_token_to_index_map(tokenized_document)

        # Get a field that, for every document token, has the first index within
        # the document that token appears
        fields['document_token_first_indices'] = \
            get_first_indices_field(tokenized_document, document_token_to_indices)

        # Setup the summary field, if it exists
        if summary is not None:
            tokenized_summary = self.summary_tokenizer.tokenize(summary)
            if self.max_summary_length is not None:
                tokenized_summary = tokenized_summary[:self.max_summary_length]
            summary_field = TextField(tokenized_summary, self.summary_token_indexers)
            fields['summary'] = summary_field

            summary_token_document_indices_field, mask_field = \
                get_token_mapping_field(document_token_to_indices, tokenized_summary)
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
