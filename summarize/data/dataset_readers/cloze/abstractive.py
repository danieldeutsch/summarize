from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, MetadataField, TextField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from overrides import overrides
from typing import Dict, Iterable, List, Optional

from summarize.data.io import JsonlReader
from summarize.data.paragraph_tokenizers import ParagraphTokenizer, ParagraphWordTokenizer


@DatasetReader.register('cloze-abstractive')
class AbstractiveClozeDatasetReader(DatasetReader):
    """
    Reads a generic single-document cloze dataset for an abstractive
    summarization model.

    Parameters
    ----------
    document_token_indexers: optional (default = ``{'tokens': SingleIdTokenIndexer()}``).
        The token indexers used for the document tokens.
    topic_token_indexers: optional (default = ``document_token_indexers``).
        The token indexers used for the topic tokens.
    context_token_indexers: optional (default = ``cloze_token_indexers``).
        The token indexers used for the context tokens.
    cloze_token_indexers: optional (default = ``document_token_indexers``).
        The token indexers used for the cloze tokens.
    document_tokenizer: optional (default = ``ParagraphWordTokenizer``).
        The tokenizer for the document text.
    topic_tokenizer: optional (default = ``WordTokenizer``).
        The tokenizer for the topics.
    context_tokenizer: optional (default = ``document_tokenizer``).
        The tokenizer for the context.
    cloze_tokenizer: optional (default = ``WordTokenizer``).
        The tokenizer for the cloze.
    max_document_length: ``int``, optional (default = ``None``).
        The maximum number of document tokens to use. If ``None``, no truncation is performed. The
        truncation runs after the tokenization step, so this length number includes any ``start_tokens``,
        ``end_tokens``, etc. It does not ensure that the ``end_tokens`` will still be at the end
        of the sequence.
    max_context_length: ``int``, optional (default = ``None``).
        The maximum number of content tokens to use. If not ``None``, the last
        ``max_content_length`` tokens of the context are used.
    max_cloze_length: ``int``, optional (default = ``None``).
        The maximum number of cloze tokens to use.
    """
    def __init__(self,
                 document_token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 topic_token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 context_token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 cloze_token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 document_tokenizer: Optional[ParagraphTokenizer] = None,
                 topic_tokenizer: Optional[Tokenizer] = None,
                 context_tokenizer: Optional[ParagraphTokenizer] = None,
                 cloze_tokenizer: Optional[Tokenizer] = None,
                 max_document_length: Optional[int] = None,
                 max_context_length: Optional[int] = None,
                 max_cloze_length: Optional[int] = None,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.document_token_indexers = document_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.topic_token_indexers = topic_token_indexers or self.document_token_indexers
        self.cloze_token_indexers = cloze_token_indexers or self.document_token_indexers
        self.context_token_indexers = context_token_indexers or self.cloze_token_indexers
        self.document_tokenizer = document_tokenizer or ParagraphWordTokenizer()
        self.topic_tokenizer = topic_tokenizer or WordTokenizer()
        self.context_tokenizer = context_tokenizer or self.document_tokenizer
        self.cloze_tokenizer = cloze_tokenizer or WordTokenizer()
        self.max_document_length = max_document_length
        self.max_context_length = max_context_length
        self.max_cloze_length = max_cloze_length

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        with JsonlReader(file_path) as f:
            for instance in f:
                document = instance['document']
                topics = instance['topics']
                context = instance['context']
                cloze = instance['cloze']
                yield self.text_to_instance(document, topics, context, cloze=cloze)

    @overrides
    def text_to_instance(self,
                         document: List[str],
                         topics: List[str],
                         context: List[str],
                         cloze: Optional[str] = None) -> Instance:
        """
        Parameters
        ----------
        document:
            The list of document sentences.
        topics:
            The list of topics.
        context:
            The list of context sentences.
        cloze:
            The cloze string.
        """
        fields = {}

        # Setup the document field
        tokenized_document = self.document_tokenizer.tokenize(document)
        if self.max_document_length is not None:
            tokenized_document = tokenized_document[:self.max_document_length]
        fields['document'] = TextField(tokenized_document, self.document_token_indexers)

        # Setup the topics
        tokenized_topics = [self.topic_tokenizer.tokenize(topic) for topic in topics]
        topic_fields = [TextField(tokenized_topic, self.topic_token_indexers) for tokenized_topic in tokenized_topics]
        fields['topics'] = ListField(topic_fields)

        # Setup the context
        tokenized_context = self.context_tokenizer.tokenize(context)
        if self.max_context_length is not None:
            # We take the last tokens instead of the first because the cloze
            # comes immediately after the context
            tokenized_context = tokenized_context[-self.max_context_length:]
        fields['context'] = TextField(tokenized_context, self.context_token_indexers)

        # Setup the cloze field, if it exists
        if cloze is not None:
            tokenized_cloze = self.cloze_tokenizer.tokenize(cloze)
            if self.max_cloze_length is not None:
                tokenized_cloze = tokenized_cloze[:self.max_cloze_length]
            fields['cloze'] = TextField(tokenized_cloze, self.cloze_token_indexers)

        # Pass the original data through as metadata
        metadata = {}
        metadata['document'] = document
        metadata['topics'] = topics
        metadata['context'] = context
        if cloze is not None:
            metadata['cloze'] = cloze
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
