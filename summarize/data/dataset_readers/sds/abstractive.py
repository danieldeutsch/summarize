from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from overrides import overrides
from typing import Dict, Iterable, List, Optional

from summarize.data.io import JsonlReader
from summarize.data.paragraph_tokenizers import ParagraphTokenizer, ParagraphWordTokenizer


@DatasetReader.register('sds-abstractive')
class AbstractiveDatasetReader(DatasetReader):
    """
    Reads a generic single-document summarization dataset for an abstractive
    summarization model. Both the document and the summary is expected to be a
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

        # Setup the document field
        tokenized_document = self.document_tokenizer.tokenize(document)
        if self.max_document_length is not None:
            tokenized_document = tokenized_document[:self.max_document_length]
        document_field = TextField(tokenized_document, self.document_token_indexers)
        fields['document'] = document_field

        # Setup the summary field, if it exists
        if summary is not None:
            tokenized_summary = self.summary_tokenizer.tokenize(summary)
            if self.max_summary_length is not None:
                tokenized_summary = tokenized_summary[:self.max_summary_length]
            summary_field = TextField(tokenized_summary, self.summary_token_indexers)
            fields['summary'] = summary_field

        # Pass the original data through as metadata
        metadata = {}
        metadata['document'] = document
        if summary is not None:
            metadata['summary'] = summary
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
