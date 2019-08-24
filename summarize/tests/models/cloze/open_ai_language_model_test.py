import os
import pytest
import unittest

from summarize.common.testing import FIXTURES_ROOT
from summarize.data.io import JsonlReader
from summarize.models.cloze import OpenAILanguageModel

_MODEL_DIR = 'experiments/deutsch2019/baselines/open-ai/models/345M'


class TestOpenAILanguageModel(unittest.TestCase):
    @pytest.mark.skip(reason='Too slow')
    @pytest.mark.skipif(not os.path.exists(_MODEL_DIR), reason='OpenAI Language Model does not exist')
    def test_open_ai_language_model(self):
        """
        Tests to make sure the OpenAI language model successfully loads and
        can process data.
        """
        length = 100
        temperature = 1.0
        top_k = 20
        lm = OpenAILanguageModel(_MODEL_DIR, length, temperature, top_k)

        # This can be quite slow, so we only do it for 1 instance
        with JsonlReader(f'{FIXTURES_ROOT}/data/cloze.jsonl') as f:
            for instance in f:
                context = instance['context']
                input_text = ' '.join(context)
                sentence = lm.sample_next_sentence(input_text)
                assert sentence is not None
                break
