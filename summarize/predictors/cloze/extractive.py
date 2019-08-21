import json
from allennlp.common.util import JsonDict
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('cloze-extractive-predictor')
class ClozeExtractivePredictor(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        indices = outputs['predicted_indices']
        document = outputs['metadata']['document']
        cloze = [document[index] for index in indices]
        output_data = {'cloze': cloze}
        return json.dumps(output_data) + '\n'
