from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_filter import WordFilter, PassThroughWordFilter
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.word_stemmer import WordStemmer, PassThroughWordStemmer
from overrides import overrides
from typing import List

from summarize.data.paragraph_tokenizers import ParagraphTokenizer


@ParagraphTokenizer.register('word')
class ParagraphWordTokenizer(ParagraphTokenizer):
    """
    A ``ParagraphWordTokenizer`` is a wrapper around the ``WordTokenizer`` at the
    paragraph-level. It includes the ability to insert tokens in between the
    sentence tokens.

    Parameters
    ----------
    word_splitter: ``WordSplitter``, optional (default = ``None``)
        See ``WordTokenizer``
    word_filter: ``WordFilter``, optional (default = ``PassThroughWordFilter()``)
        See ``WordTokenizer``
    word_stemmer: ``WordStemmer``, optional (default = ``PassThroughWordStemmer()``)
        See ``WordTokenizer``
    start_tokens: ``List[str]``, optional (default = ``[]``)
        See ``WordTokenizer``
    end_tokens: ``List[str]``, optional (default = ``[]``)
        See ``WordTokenizer``
    in_between_tokens: ``List[str]``, optional (default = ``[]``)
        The tokens to insert in between sentences.
    """
    def __init__(self,
                 word_splitter: WordSplitter = None,
                 word_filter: WordFilter = PassThroughWordFilter(),
                 word_stemmer: WordStemmer = PassThroughWordStemmer(),
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 in_between_tokens: List[str] = None):
        self.tokenizer = WordTokenizer(word_splitter=word_splitter,
                                       word_filter=word_filter,
                                       word_stemmer=word_stemmer)
        self.start_tokens = start_tokens or []
        self.start_tokens = [Token(token) for token in self.start_tokens]
        self.end_tokens = end_tokens or []
        self.end_tokens = [Token(token) for token in self.end_tokens]
        self.in_between_tokens = in_between_tokens or []
        self.in_between_tokens = [Token(token) for token in self.in_between_tokens]

    @overrides
    def tokenize(self, texts: List[str]) -> List[Token]:
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        tokens = []
        if self.start_tokens:
            tokens.extend(self.start_tokens)
        for i, tokenized_text in enumerate(tokenized_texts):
            tokens.extend(tokenized_text)

            # Add the in-between tokens if this is not the last sentence
            if i != len(tokenized_texts) - 1:
                tokens.extend(self.in_between_tokens)
        if self.end_tokens:
            tokens.extend(self.end_tokens)
        return tokens
