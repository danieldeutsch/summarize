import spacy
import unittest
from nltk import word_tokenize

from summarize.data.dataset_setup.tokenize import tokenize


class TestTokenize(unittest.TestCase):
    def test_spacy_tokenize(self):
        nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])
        field = "Hi, I'm Dan."
        expected = "Hi , I 'm Dan ."
        actual = tokenize(nlp, field)
        assert expected == actual

        field = [['The first.', 'The second.'], 'The third.']
        expected = [['The first .', 'The second .'], 'The third .']
        actual = tokenize(nlp, field)
        assert expected == actual

    def test_nltk_tokenize(self):
        field = "Hi, I'm Dan."
        expected = "Hi , I 'm Dan ."
        actual = tokenize(word_tokenize, field)
        assert expected == actual
