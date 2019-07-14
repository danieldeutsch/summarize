import unittest

from summarize.data.paragraph_tokenizers import ParagraphWordTokenizer


class TestParagraphWordTokenizer(unittest.TestCase):
    def test_in_between_tokens(self):
        texts = [
            'This is the first sentence.',
            'Followed by the second.',
            'And the third!'
        ]

        tokenizer = ParagraphWordTokenizer()
        expected = [
            'This', 'is', 'the', 'first', 'sentence', '.',
            'Followed', 'by', 'the', 'second', '.',
            'And', 'the', 'third', '!'
        ]
        tokens = tokenizer.tokenize(texts)
        actual = list(map(str, tokens))
        assert expected == actual

        tokenizer = ParagraphWordTokenizer(start_tokens=['@start@'],
                                           end_tokens=['@end@'],
                                           in_between_tokens=['</s>', '<s>'])
        expected = [
            '@start@', 'This', 'is', 'the', 'first', 'sentence', '.', '</s>', '<s>',
            'Followed', 'by', 'the', 'second', '.', '</s>', '<s>',
            'And', 'the', 'third', '!', '@end@'
        ]
        tokens = tokenizer.tokenize(texts)
        actual = list(map(str, tokens))
        assert expected == actual
