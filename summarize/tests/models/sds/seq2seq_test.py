from allennlp.common.testing import ModelTestCase

# Some imports necessary in order to register the dataset reader and model
import summarize.data.dataset_readers.sds
import summarize.models.sds
from summarize.common.testing import FIXTURES_ROOT


class Seq2SeqModelTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(f'{FIXTURES_ROOT}/configs/sds/seq2seq.jsonnet',
                          f'{FIXTURES_ROOT}/data/sds.jsonl')

    def test_sds_seq2seq_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()