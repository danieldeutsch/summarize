import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ArrayField, ListField, MetadataField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides
from typing import Dict, Iterable, List, Optional

from summarize.data.io import JsonlReader


@DatasetReader.register('cloze-extractive')
class ExtractiveClozeDatasetReader(DatasetReader):
    """
    The ExtractiveClozeDatasetReader reads in a generic cloze dataset for an
    extractive summarization model.

    Parameters
    ----------
    tokenizer:
        The tokenizer that will be used to tokenize the document, topic, and context.
    token_indexers:
        The token indexers for the document, topic, and context.
    max_num_sentences:
        The maximum number of document sentences to take. If a sentence with a
        corresponding positive label in the ground-truth is removed, the label
        is also removed. If None, all will be taken.
    max_sentence_length:
        The maximum number of sentence tokens to take (after tokenization). If None,
        all will be taken.
    max_context_length:
        The maximum context length after tokenization. If None, all will be taken.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_num_sentences: int = None,
                 max_sentence_length: int = None,
                 max_context_length: int = None,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_num_sentences = max_num_sentences
        self.max_sentence_length = max_sentence_length
        self.max_context_length = max_context_length

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        with JsonlReader(file_path) as f:
            for instance in f:
                document = instance['document']
                topics = instance['topics']
                context = instance['context']
                cloze = instance['cloze']
                labels = instance['labels']
                yield self.text_to_instance(document, topics, context, labels=labels, cloze=cloze)

    def text_to_instance(self,
                         document: List[str],
                         topics: List[str],
                         context: List[str],
                         cloze: Optional[str] = None,
                         labels: Optional[List[int]] = None) -> Instance:
        """
        Parameters
        ----------
        document:
            A sentence-tokenized input document.
        topics:
            The list of topics.
        context:
            The sentence-tokenized context for the cloze.
        cloze:
            The cloze string.
        labels:
            The indices of the positive labels in the extractive ground-truth.
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

        # Setup the topics
        tokenized_topics = [self.tokenizer.tokenize(topic) for topic in topics]
        topics_text_fields = [TextField(tokens, self.token_indexers) for tokens in tokenized_topics]
        topics_field = ListField(topics_text_fields)
        fields['topics'] = topics_field

        # Setup the context
        context = ' '.join(context)
        tokenized_context = self.tokenizer.tokenize(context)
        if self.max_context_length is not None:
            # We take the last tokens in stead of the first because the cloze
            # comes immediately after the context
            tokenized_context = tokenized_context[-self.max_context_length:]
        context_field = TextField(tokenized_context, self.token_indexers)
        fields['context'] = context_field

        if labels is not None:
            label_array = np.zeros(len(document), dtype=np.uint8)
            label_array[labels] = 1
            label_field = ArrayField(label_array)
            fields['labels'] = label_field

        metadata = {}
        metadata['document'] = document
        metadata['topics'] = topics
        metadata['context'] = context
        if cloze is not None:
            metadata['cloze'] = cloze
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
