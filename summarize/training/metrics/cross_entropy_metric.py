import numpy as np
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Dict


@Metric.register('cross-entropy')
class CrossEntropyMetric(Metric):
    def __init__(self) -> None:
        self.total_loss = 0
        self.total_num_tokens = 0

    @overrides
    def __call__(self, loss: float, num_tokens: int) -> None:
        self.total_loss += loss
        self.total_num_tokens += num_tokens

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        cross_entropy = self.total_loss / self.total_num_tokens
        perplexity = np.exp(cross_entropy)
        if reset:
            self.total_loss = 0
            self.total_num_tokens = 0
        return {
            'cross-entropy': cross_entropy,
            'perplexity': perplexity
        }
