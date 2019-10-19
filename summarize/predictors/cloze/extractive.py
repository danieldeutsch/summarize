import json
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('cloze-extractive-predictor')
class ClozeExtractivePredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        document = json_dict['document']
        topics = json_dict['topics']
        context = json_dict['context']
        return self._dataset_reader.text_to_instance(document=document,
                                                     topics=topics,
                                                     context=context)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        indices = outputs['predicted_indices']
        document = outputs['metadata']['document']
        cloze = [document[index] for index in indices]
        output_data = {'cloze': cloze}
        return json.dumps(output_data) + '\n'
