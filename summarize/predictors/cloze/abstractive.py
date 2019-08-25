import json
from allennlp.common.util import JsonDict
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('cloze-abstractive-predictor')
class ClozeAbstractivePredictor(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        cloze = outputs['cloze']
        output_data = {'cloze': cloze}
        return json.dumps(output_data) + '\n'
