# pylint: disable=invalid-name

from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary

from summarize.nn.beam_search import BeamSearch
from summarize.nn.beam_search.beam_search import StepFunctionType
from summarize.nn.beam_search.length_penalizers import WuLengthPenalizer


transition_probabilities = torch.tensor(  # pylint: disable=not-callable
        [[0.0, 0.4, 0.3, 0.2, 0.1, 0.0],  # start token -> jth token
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 1st token -> jth token
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 2nd token -> jth token
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # ...
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # ...
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # end token -> jth token
)

# A set of transition probabilities that can loop on the 4th token until
# emitting the end token
infinite_transition_probabilities = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.1, 0.9],  # start token -> jth token
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 1st token -> jth token
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2nd token -> jth token
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ...
         [0.0, 0.0, 0.0, 0.0, 0.1, 0.9],  # ...
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # end token -> jth token
)


def take_step(last_predictions: torch.Tensor,
              state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Take decoding step.

    This is a simple function that defines how probabilities are computed for the
    next time step during the beam search.

    We use a simple target vocabulary of size 6. In this vocabulary, index 0 represents
    the start token, and index 5 represents the end token. The transition probability
    from a state where the last predicted token was token `j` to new token `i` is
    given by the `(i, j)` element of the matrix `transition_probabilities`.
    """
    log_probs_list = []
    for last_token in last_predictions:
        log_probs = torch.log(transition_probabilities[last_token.item()])
        log_probs_list.append(log_probs)

    return torch.stack(log_probs_list), state


def take_infinite_step(last_predictions: torch.Tensor,
                       state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    log_probs_list = []
    for last_token in last_predictions:
        log_probs = torch.log(infinite_transition_probabilities[last_token.item()])
        log_probs_list.append(log_probs)

    return torch.stack(log_probs_list), state


class BeamSearchTest(AllenNlpTestCase):

    def setUp(self):
        super(BeamSearchTest, self).setUp()
        self.vocab = Vocabulary(non_padded_namespaces=['tokens'])
        for i in range(transition_probabilities.size(0)):
            self.vocab.add_token_to_namespace(str(i))
        self.end_symbol = str(transition_probabilities.size()[0] - 1)
        self.end_index = transition_probabilities.size()[0] - 1
        # Ensure the end symbol has the expected index
        assert self.end_index == self.vocab.get_token_index(self.end_symbol)
        self.beam_search = BeamSearch(self.vocab, beam_size=3, end_symbol=self.end_symbol,
                                      max_steps=10)

        # This is what the top k should look like for each item in the batch.
        self.expected_top_k = np.array(
                [[1, 2, 3, 4, 5],
                 [2, 3, 4, 5, 5],
                 [3, 4, 5, 5, 5]]
        )

        # This is what the log probs should look like for each item in the batch.
        self.expected_log_probs = np.log(np.array([0.4, 0.3, 0.2]))  # pylint: disable=assignment-from-no-return

    def _check_results(self,
                       batch_size: int = 5,
                       expected_top_k: np.array = None,
                       expected_log_probs: np.array = None,
                       beam_search: BeamSearch = None,
                       state: Dict[str, torch.Tensor] = None,
                       step: StepFunctionType = None,
                       rtol: float = 1e-7) -> None:
        expected_top_k = expected_top_k if expected_top_k is not None else self.expected_top_k
        expected_log_probs = expected_log_probs if expected_log_probs is not None else self.expected_log_probs
        state = state or {}
        step = step or take_step

        beam_search = beam_search or self.beam_search
        beam_size = beam_search.beam_size

        initial_predictions = torch.tensor([0] * batch_size)  # pylint: disable=not-callable
        top_k, log_probs = beam_search.search(initial_predictions, state, step)  # type: ignore

        # top_k should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(top_k.size())[:-1] == [batch_size, beam_size]
        np.testing.assert_array_equal(top_k[0].numpy(), expected_top_k)

        # log_probs should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(log_probs.size()) == [batch_size, beam_size]
        np.testing.assert_allclose(log_probs[0].numpy(), expected_log_probs, rtol=rtol)

    def test_search(self):
        self._check_results()

    def test_finished_state(self):
        state = {}
        state["foo"] = torch.tensor(  # pylint: disable=not-callable
                [[1, 0, 1],
                 [2, 0, 1],
                 [0, 0, 1],
                 [1, 1, 1],
                 [0, 0, 0]]
        )
        # shape: (batch_size, 3)

        expected_finished_state = {}
        expected_finished_state["foo"] = np.array(
                [[1, 0, 1],
                 [1, 0, 1],
                 [1, 0, 1],
                 [2, 0, 1],
                 [2, 0, 1],
                 [2, 0, 1],
                 [0, 0, 1],
                 [0, 0, 1],
                 [0, 0, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
        )
        # shape: (batch_size x beam_size, 3)

        self._check_results(state=state)

        # check finished state.
        for key, array in expected_finished_state.items():
            np.testing.assert_allclose(state[key].numpy(), array)

    def test_batch_size_of_one(self):
        self._check_results(batch_size=1)

    def test_greedy_search(self):
        beam_search = BeamSearch(self.vocab, beam_size=1, end_symbol=self.end_symbol)
        expected_top_k = np.array([[1, 2, 3, 4, 5]])
        expected_log_probs = np.log(np.array([0.4]))  # pylint: disable=assignment-from-no-return
        self._check_results(expected_top_k=expected_top_k,
                            expected_log_probs=expected_log_probs,
                            beam_search=beam_search)

    def test_early_stopping(self):
        """
        Checks case where beam search will reach `max_steps` before finding end tokens.
        """
        beam_search = BeamSearch(self.vocab, beam_size=3, end_symbol=self.end_symbol, max_steps=3)
        expected_top_k = np.array(
                [[1, 2, 3],
                 [2, 3, 4],
                 [3, 4, 5]]
        )
        expected_log_probs = np.log(np.array([0.4, 0.3, 0.2]))  # pylint: disable=assignment-from-no-return
        self._check_results(expected_top_k=expected_top_k,
                            expected_log_probs=expected_log_probs,
                            beam_search=beam_search)

    def test_different_per_node_beam_size(self):
        # per_node_beam_size = 1
        beam_search = BeamSearch(self.vocab, beam_size=3, end_symbol=self.end_symbol, per_node_beam_size=1)
        self._check_results(beam_search=beam_search)

        # per_node_beam_size = 2
        beam_search = BeamSearch(self.vocab, beam_size=3, end_symbol=self.end_symbol, per_node_beam_size=2)
        self._check_results(beam_search=beam_search)

    def test_catch_bad_config(self):
        """
        If `per_node_beam_size` (which defaults to `beam_size`) is larger than
        the size of the target vocabulary, `BeamSearch.search` should raise
        a ConfigurationError.
        """
        beam_search = BeamSearch(self.vocab, beam_size=20, end_symbol=self.end_symbol)
        with pytest.raises(ConfigurationError):
            self._check_results(beam_search=beam_search)

    def test_warn_for_bad_log_probs(self):
        # The only valid next step from the initial predictions is the end index.
        # But with a beam size of 3, the call to `topk` to find the 3 most likely
        # next beams will result in 2 new beams that are invalid, in that have probability of 0.
        # The beam search should warn us of this.
        initial_predictions = torch.LongTensor([self.end_index-1, self.end_index-1])
        with pytest.warns(RuntimeWarning, match="Infinite log probabilities"):
            self.beam_search.search(initial_predictions, {}, take_step)

    def test_empty_sequences(self):
        initial_predictions = torch.LongTensor([self.end_index-1, self.end_index-1])
        beam_search = BeamSearch(self.vocab, beam_size=1, end_symbol=self.end_symbol)
        with pytest.warns(RuntimeWarning, match="Empty sequences predicted"):
            predictions, log_probs = beam_search.search(initial_predictions, {}, take_step)
        # predictions hould have shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(predictions.size()) == [2, 1, 1]
        # log probs hould have shape `(batch_size, beam_size)`.
        assert list(log_probs.size()) == [2, 1]
        assert (predictions == self.end_index).all()
        assert (log_probs == 0).all()

    def test_min_steps(self):
        # Ensure without a minimum length that the end token is directly emitted.
        beam_search = BeamSearch(self.vocab, beam_size=1, end_symbol=self.end_symbol)
        expected_top_k = np.array([[5]])
        expected_log_probs = np.log(np.array([0.9], dtype=np.float32))  # pylint: disable=assignment-from-no-return
        with pytest.warns(RuntimeWarning, match="Empty sequences predicted"):
            self._check_results(expected_top_k=expected_top_k,
                                expected_log_probs=expected_log_probs,
                                beam_search=beam_search,
                                step=take_infinite_step)

        # Test min_steps = 1, which will not force the search code into
        # the for loop
        beam_search = BeamSearch(self.vocab, beam_size=1, end_symbol=self.end_symbol, min_steps=1)
        expected_top_k = np.array([[4, 5]])
        expected_log_probs = np.log(np.array([0.09]))  # pylint: disable=assignment-from-no-return
        self._check_results(expected_top_k=expected_top_k,
                            expected_log_probs=expected_log_probs,
                            beam_search=beam_search,
                            step=take_infinite_step)

        # Test min_steps > 1, so the for loop code will run
        beam_search = BeamSearch(self.vocab, beam_size=1, end_symbol=self.end_symbol, min_steps=3)
        expected_top_k = np.array([[4, 4, 4, 5]])
        expected_log_probs = np.log(np.array([0.0009]))  # pylint: disable=assignment-from-no-return
        self._check_results(expected_top_k=expected_top_k,
                            expected_log_probs=expected_log_probs,
                            beam_search=beam_search,
                            step=take_infinite_step)

    def test_min_steps_warn_for_bad_log_probs(self):
        initial_predictions = torch.LongTensor([0] * 2)
        beam_search = BeamSearch(self.vocab, beam_size=1, end_symbol=self.end_symbol, min_steps=5)
        with pytest.warns(RuntimeWarning, match="Infinite log probabilities"):
            beam_search.search(initial_predictions, {}, take_step)

    def test_reconstruct_predictions(self):
        predictions = [
            torch.LongTensor([
                [0, 1, 2],
                [3, 4, 5]
            ]),
            torch.LongTensor([
                [6, 7, 8],
                [9, 10, 11]
            ]),
            torch.LongTensor([
                [12, 13, 14],
                [15, 16, 17]
            ]),
            torch.LongTensor([
                [18, 19, 20],
                [21, 22, 23]
            ])
        ]
        backpointers = [
            torch.LongTensor([
                [0, 1, 0],
                [0, 1, 2]
            ]),
            torch.LongTensor([
                [2, 1, 0],
                [0, 0, 1]
            ]),
            torch.LongTensor([
                [1, 2, 0],
                [0, 1, 2]
            ])
        ]
        reconstructed_predictions = torch.LongTensor([
            [
                [1, 7, 13, 18],
                [0, 6, 14, 19],
                [0, 8, 12, 20]
            ],
            [
                [3, 9, 15, 21],
                [3, 9, 16, 22],
                [4, 10, 17, 23]
            ]
        ])

        actual_reconstructed = self.beam_search._reconstruct_predictions(predictions, backpointers)
        assert torch.equal(reconstructed_predictions, actual_reconstructed)

        for num_steps in range(1, 5):
            actual_reconstructed = \
                self.beam_search._reconstruct_predictions(predictions, backpointers, num_steps=num_steps)
            assert torch.equal(reconstructed_predictions[:, :, -num_steps:], actual_reconstructed)

    def test_disallow_repeated_ngrams_errors(self):
        # Inconsistent disallowed length and exception length
        with pytest.raises(Exception):
            BeamSearch(self.vocab, beam_size=2, max_steps=10, end_symbol=self.end_symbol,
                       disallow_repeated_ngrams=2, repeated_ngrams_exceptions=[["3", "3", "3"]])

        # Token not present in vocabulary
        with pytest.raises(Exception):
            BeamSearch(self.vocab, beam_size=2, max_steps=10, end_symbol=self.end_symbol,
                       disallow_repeated_ngrams=2, repeated_ngrams_exceptions=[["A", "3"]])

    def test_length_penalizer(self):
        # This is an extreme value for the Wu penalizer just to force
        # the outputs to switch order
        length_penalizer = WuLengthPenalizer(-10)
        beam_search = BeamSearch(self.vocab, beam_size=3, end_symbol=self.end_symbol,
                                 max_steps=10, length_penalizer=length_penalizer)
        # The outputs are in the opposite order than expected
        expected_top_k = np.array(
                [[3, 4, 5, 5, 5],
                 [2, 3, 4, 5, 5],
                 [1, 2, 3, 4, 5]]
        )
        expected_log_probs = np.log(np.array([0.2, 0.3, 0.4]))
        self._check_results(expected_top_k=expected_top_k,
                            expected_log_probs=expected_log_probs,
                            beam_search=beam_search,
                            step=take_step)
