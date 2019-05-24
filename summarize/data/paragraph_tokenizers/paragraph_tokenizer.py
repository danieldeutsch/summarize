from typing import List

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token


class ParagraphTokenizer(Registrable):
    """
    A ``ParagraphTokenizer`` is a wrapper around an AllenNLP ``Tokenizer`` for tokenizing
    a list of strings into tokens. The primary use is for tokenizing a pre-sentence-split
    paragraph into a single list of tokens. Having this abstraction at the paragraph-level
    allows for additional functionality, like adding tokens in between the sentences.
    """
    def tokenize(self, texts: List[str]) -> List[Token]:
        """
        Actually implements splitting sentences into tokens.

        Returns
        -------
        tokens : ``List[Token]``
        """
        raise NotImplementedError
