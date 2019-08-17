import json
from allennlp.common.util import JsonDict
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('sds-extractive-predictor')
class ExtractivePredictor(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        indices = outputs['predicted_indices']
        document = outputs['metadata']['document']
        summary = [document[index] for index in indices]
        output_data = {'summary': summary}
        return json.dumps(output_data) + '\n'
