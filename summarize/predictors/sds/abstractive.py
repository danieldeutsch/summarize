import json
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('sds-abstractive-predictor')
class AbstractivePredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        document = json_dict['document']
        return self._dataset_reader.text_to_instance(document=document)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        summary = outputs['summary']
        output_data = {'summary': [summary]}
        return json.dumps(output_data) + '\n'
