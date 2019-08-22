from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, MetadataField, NamespaceSwappingField, TextField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from overrides import overrides
from typing import Dict, Iterable, List, Optional

from summarize.data.dataset_readers.util import get_first_indices_field, get_token_mapping_field, get_token_to_index_map
from summarize.data.io import JsonlReader
from summarize.data.paragraph_tokenizers import ParagraphTokenizer, ParagraphWordTokenizer


@DatasetReader.register('cloze-pointer-generator')
class PointerGeneratorClozeDatasetReader(DatasetReader):
    """
    Reads a generic single-document cloze dataset for a the Pointer-Generator
    cloze model. In addition to the fields provided by the ``AbstractiveClozeDatasetReader``,
    this class also returns the first index of each document token in the document,
    the document tokens indexed in the cloze vocabulary namespace, and the indices
    of each context and cloze token in the document.

    Parameters
    ----------
    document_token_indexers: optional (default = ``{'tokens': SingleIdTokenIndexer()}``).
        The token indexers used for the document tokens.
    topic_token_indexers: optional (default = ``document_token_indexers``).
        The token indexers used for the topic tokens.
    context_token_indexers: optional (default = ``document_token_indexers``).
        The token indexers used for the context tokens.
    cloze_token_indexers: optional (default = ``document_token_indexers``).
        The token indexers used for the cloze tokens.
    document_tokenizer: optional (default = ``ParagraphWordTokenizer``).
        The tokenizer for the document text.
    topic_tokenizer: optional (default = ``WordTokenizer``).
        The tokenizer for the topics.
    context_tokenizer: optional (default = ``document_tokenizer``).
        The tokenizer for the context.
    cloze_tokenizer: optional (default = ``topic_tokenizer``).
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
        self.context_token_indexers = context_token_indexers or self.document_token_indexers
        self.cloze_token_indexers = cloze_token_indexers or self.document_token_indexers
        self.document_tokenizer = document_tokenizer or ParagraphWordTokenizer()
        self.topic_tokenizer = topic_tokenizer or WordTokenizer()
        self.context_tokenizer = context_tokenizer or self.document_tokenizer
        self.cloze_tokenizer = cloze_tokenizer or self.topic_tokenizer
        self.max_document_length = max_document_length
        self.max_context_length = max_context_length
        self.max_cloze_length = max_cloze_length
        self.cloze_namespace = self.cloze_token_indexers['tokens'].namespace

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

        # There is some weirdness that can happen if the document tokens are lowercased
        # but the context/cloze tokens are not (or vice versa). We will deal with that when
        # it's necessary. For now, we don't allow it.
        assert self.document_token_indexers['tokens'].lowercase_tokens == self.cloze_token_indexers['tokens'].lowercase_tokens
        assert self.context_token_indexers['tokens'].lowercase_tokens == self.cloze_token_indexers['tokens'].lowercase_tokens
        if self.document_token_indexers['tokens'].lowercase_tokens:
            document = [sentence.lower() for sentence in document]
            context = [sentence.lower() for sentence in context]
            if cloze is not None:
                cloze = cloze.lower()

        # Setup the document field
        tokenized_document = self.document_tokenizer.tokenize(document)
        if self.max_document_length is not None:
            tokenized_document = tokenized_document[:self.max_document_length]
        fields['document'] = TextField(tokenized_document, self.document_token_indexers)

        # Get the document token indices but in the cloze namespace
        fields['document_in_cloze_namespace'] = NamespaceSwappingField(tokenized_document, self.cloze_namespace)

        # Build a map from token to all of the indices that token appears
        document_token_to_indices = get_token_to_index_map(tokenized_document)

        # Get a field that, for every document token, has the first index within
        # the document that token appears
        fields['document_token_first_indices'] = \
            get_first_indices_field(tokenized_document, document_token_to_indices)

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

        context_token_document_indices_field, mask_field = \
            get_token_mapping_field(document_token_to_indices, tokenized_context)
        fields['context_token_document_indices'] = context_token_document_indices_field
        fields['context_token_document_indices_mask'] = mask_field

        # Setup the cloze field, if it exists
        if cloze is not None:
            tokenized_cloze = self.cloze_tokenizer.tokenize(cloze)
            if self.max_cloze_length is not None:
                tokenized_cloze = tokenized_cloze[:self.max_cloze_length]
            fields['cloze'] = TextField(tokenized_cloze, self.cloze_token_indexers)

            cloze_token_document_indices_field, mask_field = \
                get_token_mapping_field(document_token_to_indices, tokenized_cloze)
            fields['cloze_token_document_indices'] = cloze_token_document_indices_field
            fields['cloze_token_document_indices_mask'] = mask_field

        # Pass the original data through as metadata
        metadata = {}
        metadata['document'] = document
        metadata['document_tokens'] = [str(token) for token in tokenized_document]
        metadata['topics'] = topics
        metadata['context'] = context
        if cloze is not None:
            metadata['cloze'] = cloze
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
