from allennlp.common.testing import ModelTestCase

# Some imports necessary in order to register the dataset reader, model, and modules
import summarize.data.dataset_readers.sds
import summarize.models.sds
import summarize.modules.matrix_attention
import summarize.training.metrics
from summarize.common.testing import FIXTURES_ROOT


class PointerGeneratorModelTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(f'{FIXTURES_ROOT}/configs/sds/pointer-generator.jsonnet',
                          f'{FIXTURES_ROOT}/data/sds.jsonl')

    def test_sds_pointer_generator_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent(keys_to_ignore='log_probabilities')
