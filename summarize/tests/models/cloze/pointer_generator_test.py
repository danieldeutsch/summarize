from allennlp.common.testing import ModelTestCase

# Some imports necessary in order to register the dataset reader, model, and modules
import summarize.data.dataset_readers.cloze
import summarize.models.cloze
import summarize.modules.matrix_attention
import summarize.training.metrics
from summarize.common.testing import FIXTURES_ROOT


class TestClozePointerGeneratorModel(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(f'{FIXTURES_ROOT}/configs/cloze/pointer-generator.jsonnet',
                          f'{FIXTURES_ROOT}/data/cloze.jsonl')

    def test_cloze_pointer_generator_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        # The log-probabilities are often unstable
        self.ensure_batch_predictions_are_consistent(keys_to_ignore='log_probabilities')
