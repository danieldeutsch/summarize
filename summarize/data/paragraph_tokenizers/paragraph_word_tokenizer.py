from allennlp.data.tokenizers import Token, WordTokenizer
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
    in_between_tokens: ``List[str]``, optional (default = ``[]``)
        The tokens to insert in between sentences.
    kwargs: optional
        The kwargs to pass to the ``WordTokenizer`` constructor.
    """
    def __init__(self,
                 in_between_tokens: List[str] = None,
                 **kwargs):
        self.tokenizer = WordTokenizer(**kwargs)
        self.in_between_tokens = in_between_tokens or []
        self.in_between_tokens = [Token(token) for token in self.in_between_tokens]

    @overrides
    def tokenize(self, texts: List[str]) -> List[Token]:
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        tokens = []
        for i, tokenized_text in enumerate(tokenized_texts):
            tokens.extend(tokenized_text)

            # Add the in-between tokens if this is not the last sentence
            if i != len(tokenized_texts) - 1:
                tokens.extend(self.in_between_tokens)

        return tokens
