import torch
from allennlp.training.metrics import F1Measure, Metric
from overrides import overrides
from typing import Dict, Optional


@Metric.register('binary-f1')
class BinaryF1Measure(F1Measure):
    """
    The BinaryF1Measure allows for computing the standard F1 metric using
    two binary vectors, the ground-truth labels and the predictions from the
    model. The original F1Measure computation would require the ground-truth
    predictions to be a (batch_size, ..., 2) binary tensor that marks the
    ground-truth class.
    """
    def __init__(self) -> None:
        super().__init__(1)

    @overrides
    def __call__(self,
                 gold_labels: torch.Tensor,
                 model_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 **kwargs):
        """
        Parameters
        ----------
        gold_labels: (batch_size, ...)
            The ground-truth binary labels
        model_labels: (batch_size, ...)
            The binary model predictions
        mask: (batch_size, ...)
            The mask
        """
        categorical_model_labels = model_labels.new_zeros(*model_labels.size(), 2)
        model_labels = model_labels.unsqueeze(-1)
        categorical_model_labels.scatter_(-1, model_labels, 1)
        super().__call__(categorical_model_labels, gold_labels, mask)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = super().get_metric(reset)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1_measure
        }
