import torch
from allennlp.data import Vocabulary
from typing import List, Tuple

from summarize.nn.beam_search import BeamSearch
from summarize.nn.beam_search.coverage_penalizers import CoveragePenalizer
from summarize.nn.beam_search.length_penalizers import LengthPenalizer


@BeamSearch.register('relaxed')
class RelaxedBeamSearch(BeamSearch):
    """
    RelaxedBeamSearch is an implementation of beam search that allows reusing
    beam entries after a completed sequence has been found and continues to search
    until either the top entry in the beam is a complete sequence or the maximum
    number of steps is reached. This implementation was done to match the beam
    search results of OpenNMT.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 beam_size: int,
                 namespace: str = 'tokens',
                 end_symbol: str = None,
                 min_steps: int = None,
                 max_steps: int = 50,
                 per_node_beam_size: int = None,
                 disallow_repeated_ngrams: int = None,
                 repeated_ngrams_exceptions: List[List[str]] = None,
                 length_penalizer: LengthPenalizer = None,
                 coverage_penalizer: CoveragePenalizer = None) -> None:
        super().__init__(vocab=vocab,
                         beam_size=beam_size,
                         namespace=namespace,
                         end_symbol=end_symbol,
                         min_steps=min_steps,
                         max_steps=max_steps,
                         per_node_beam_size=per_node_beam_size,
                         disallow_repeated_ngrams=disallow_repeated_ngrams,
                         repeated_ngrams_exceptions=repeated_ngrams_exceptions,
                         length_penalizer=length_penalizer,
                         coverage_penalizer=coverage_penalizer)

    def initialize(self, batch_size: int) -> None:
        super().initialize(batch_size)
        self._complete_predictions = [[] for _ in range(batch_size)]
        self._complete_log_probs = [[] for _ in range(batch_size)]
        self._is_finished = torch.zeros(batch_size, dtype=torch.uint8)

    def reuse_beam(self) -> bool:
        return True

    def _reconstruct_single_prediction(self, batch: int, beam: int) -> torch.Tensor:
        # TODO This is a lazy hack
        predictions = self._reconstruct_predictions(self.predictions, self.backpointers)
        return predictions[batch, beam]

    def update_state(self) -> None:
        finished = self.predictions[-1] == self._end_index
        for batch, beam in finished.nonzero():
            batch, beam = batch.item(), beam.item()
            if self._is_finished[batch]:
                continue
            if beam == 0:
                self._is_finished[batch] = 1
            prediction = self._reconstruct_single_prediction(batch, beam)
            self._complete_predictions[batch].append(prediction)
            self._complete_log_probs[batch].append(self.log_probs[-1][batch, beam])

    def is_finished(self) -> bool:
        return self._is_finished.all()

    def get_final_predictions(self) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
        final_predictions = []
        final_log_probs = []
        final_lengths = []
        for predictions, log_probs in zip(self._complete_predictions, self._complete_log_probs):
            log_probs = torch.stack(log_probs, dim=0)
            sorted_indices = torch.argsort(log_probs, descending=True)
            final_predictions.append([predictions[j.item()] for j in sorted_indices])
            final_log_probs.append(log_probs[sorted_indices])
            # - 1 because we aren't counting the end token
            final_lengths.append(torch.LongTensor([len(predictions[j.item()]) - 1 for j in sorted_indices]))

        return final_predictions, final_log_probs, final_lengths
