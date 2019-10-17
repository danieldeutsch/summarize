import torch
import warnings
from allennlp.common import Registrable, Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL
from allennlp.data import Vocabulary
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from summarize.nn.beam_search.coverage_penalizers import CoveragePenalizer
from summarize.nn.beam_search.length_penalizers import LengthPenalizer


StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name


class BeamSearch(Registrable):
    """
    Implements the beam search algorithm for decoding the most likely sequences.

    Parameters
    ----------
    beam_size : ``int``
        The width of the beam used.
    namespace : ``str``, optional (default = ``tokens``)
        The vocabulary namespace of the output symbols.
    end_symbol : ``str``, optional (default = ``END_SYMBOL``)
        The symbol of the "stop" or "end" token in the target vocabulary.
    min_steps : ``int``, optional (default = ``None``)
        The minimum number of decoding steps to take, i.e. the minimum length
        of the predicted sequences. This does not include the start or end tokens.
        No minimum is enforced if ``None``
    max_steps : ``int``, optional (default = 50)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to ``beam_size``. Setting this parameter
        to a number smaller than ``beam_size`` may give better results, as it can introduce
        more diversity into the search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <http://arxiv.org/abs/1702.01806>`_.
    disallow_repeated_ngrams: ``int``, optional (default = ``None``)
        The order of n-gram to prevent repetitions of. If ``None``, any order n-gram
        can be repeated.
    repeated_ngrams_exceptions: ``List[str]``, optional (default = ``None``)
        A list of unigrams which are exceptions to the disallowed repeated ngrams.
    length_penalizer: ``LengthPenalizer``, optional (default = ``None``)
        The length penalizer that should be used to rerank the candidate summaries
        after beam search has finished.
    coverage_penalizer: ``CoveragePenalizer``, optional (defautl = ``None``)
        The coverage penalizer that should be used to augment the scores of
        each prediction at each step of decoding. The sequence log-probabilities
        will be augmented by this score only for selecting the next token.
    """
    default_implementation = 'standard'

    def __init__(self,
                 vocab: Vocabulary,
                 beam_size: int,
                 namespace: str = 'tokens',
                 end_symbol: str = None,
                 min_steps: int = None,
                 max_steps: int = 50,
                 per_node_beam_size: int = None,
                 disallow_repeated_ngrams: int = None,
                 repeated_ngrams_exceptions: List[str] = None,
                 length_penalizer: LengthPenalizer = None,
                 coverage_penalizer: CoveragePenalizer = None) -> None:
        self.beam_size = beam_size
        end_symbol = end_symbol or END_SYMBOL
        self._end_index = vocab.get_token_index(end_symbol, namespace)
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.length_penalizer = length_penalizer
        self.coverage_penalizer = coverage_penalizer

        # Convert the token exceptions to their indexes
        self.disallow_repeated_ngrams = disallow_repeated_ngrams
        self.repeated_ngrams_exceptions = set()
        repeated_ngrams_exceptions = repeated_ngrams_exceptions or []
        token_to_index = vocab.get_token_to_index_vocabulary(namespace)
        for token in repeated_ngrams_exceptions:
            if token not in token_to_index:
                raise Exception(f'Could not add token exception {token} because {token} is not in the vocabulary')
            self.repeated_ngrams_exceptions.add(token_to_index[token])

    def _reconstruct_predictions(self,
                                 predictions: List[torch.Tensor],
                                 backpointers: List[torch.Tensor],
                                 num_steps: int = None) -> torch.Tensor:
        """
        Reconstruct the predictions using the predictions thus far and the
        backpointers for a given number of steps. The ``predictions`` can't
        just be directly traversed without the backpointers because it's the
        output of the beam search. It's impossible to know which predictions
        are still valid without the back pointers.

        Parameters
        ----------
        predictions: List[torch.Tensor]
            The list of (batch_size, beam_size) output tokens at each step of
            the search thus far
        backpointers: List[torch.Tensor]
            The list of (batch_size, beam_size) backpointers for each step of
            the search thus far.
        num_steps: int, optional (default = ``None``)
            The number of steps from the most recent to reconstruct. If ``None``,
            the entire sequence will be reconstructed.

        Returns
        -------
        torch.Tensor, (batch_size, beam_size, num_steps)
            The reconstructed predictions.
        """
        num_steps = num_steps or len(predictions)
        if num_steps == 1:
            # shape: (batch_size, beam_size, 1)
            return predictions[-1].unsqueeze(2)

        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        # The inclusive last index of the prediction
        end = len(predictions) - 1
        # The inclusive first index of the prediction
        start = end - num_steps + 1

        for timestep in range(end - 1, start, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[start].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        return all_predictions

    def _mask_disallowed_ngrams(self, log_probs: torch.Tensor) -> None:
        if self.disallow_repeated_ngrams is None or len(self.predictions) <= self.disallow_repeated_ngrams:
            return

        predictions = self._reconstruct_predictions(self.predictions, self.backpointers)
        batch_size, beam_size, num_tokens = predictions.size()
        for i in range(batch_size):
            for j in range(beam_size):
                ngrams = set()
                found_repeated = False
                for start in range(0, num_tokens - self.disallow_repeated_ngrams + 1):
                    ngram = predictions[i, j, start:start + self.disallow_repeated_ngrams].tolist()
                    if set(ngram) & self.repeated_ngrams_exceptions:
                        continue
                    ngram = tuple(ngram)
                    if ngram in ngrams:
                        found_repeated = True
                        break
                    ngrams.add(ngram)

                if found_repeated:
                    log_probs[i, j] = float('-inf')

    def _apply_length_penalty(self,
                              final_predictions: List[List[torch.Tensor]],
                              final_log_probs: List[torch.Tensor],
                              lengths: List[torch.Tensor]) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
        if self.length_penalizer is not None:
            batch_size = len(final_predictions)
            for i in range(batch_size):
                # shape: (beam_size)
                length_penalties = self.length_penalizer(lengths[i] + 2)
                # shape: (beam_size)
                penalized_scores = final_log_probs[i] / length_penalties
                # Sort the new scores in descending order
                # shape: (beam_size)
                sorted_indices = torch.argsort(penalized_scores, dim=0, descending=True)
                # Reorder the probabilities
                final_log_probs[i] = final_log_probs[i][sorted_indices]
                # Reorder the predictions
                if isinstance(final_predictions[i], torch.Tensor):
                    final_predictions[i] = final_predictions[i][sorted_indices]
                else:
                    final_predictions[i] = [final_predictions[i][index.item()] for index in sorted_indices]

    def initialize(self, batch_size: int) -> None:
        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        self.predictions: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        self.backpointers: List[torch.Tensor] = []

        # Maintains the log-probabilities of the sequences which correspond
        # to the predictions
        self.log_probs: List[torch.Tensor] = []

        # Dictionaries which map from the prefix of an ngram (the n-1 tokens
        # represented as a tuple of indicies) to the list of indices which are not
        # allowed to appear next. The lists are indexed by batch then beam index.
        self.disallowed_ngrams = [[defaultdict(list) for _ in range(self.beam_size)] for _ in range(batch_size)]

    def search(self,
               start_predictions: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.

        Notes
        -----
        If your step function returns ``-inf`` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have ``-inf`` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from ``search``
        and potentially discard sequences with non-finite log probability.

        Parameters
        ----------
        start_predictions : ``torch.Tensor``
            A tensor containing the initial predictions with shape ``(batch_size,)``.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.
        start_state : ``StateType``
            The initial state passed to the ``step`` function. Each value of the state dict
            should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
            number of dimensions.
        step : ``StepFunctionType``
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``. The beam position is the order
            of the "best" predictions according to the search (regardless of the
            order of the log-probabilities) because the output order takes into
            account the length penalty.
        """
        batch_size = start_predictions.size()[0]

        self.initialize(batch_size)

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state = step(start_predictions, start_state)

        # If a minimum number of steps is required, we cannot directly emit
        # the end index.
        if self.min_steps is not None and self.min_steps > 0:
            start_class_log_probabilities[:, self._end_index] = float('-inf')

        num_classes = start_class_log_probabilities.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise ConfigurationError(f"Target vocab size ({num_classes:d}) too small "
                                     f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                                     f"Please decrease beam_size or per_node_beam_size.")

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probabilities, start_predicted_classes = \
            start_class_log_probabilities.topk(self.beam_size)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn("Empty sequences predicted. You may want to increase the beam size or ensure "
                          "your step function is working properly.",
                          RuntimeWarning)
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        self.predictions.append(start_predicted_classes)

        self.log_probs.append(last_log_probabilities)

        self.update_state()

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes),
            float("-inf")
        )
        if not self.reuse_beam():
            log_probs_after_end[:, self._end_index] = 0.

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor.\
                unsqueeze(1).\
                expand(batch_size, self.beam_size, *last_dims).\
                reshape(batch_size * self.beam_size, *last_dims)

        # Maintains the length of each prediction, not including the start
        # or end token
        self.lengths = start_class_log_probabilities.new_zeros(batch_size * self.beam_size, dtype=torch.long)

        # If we are applying a coverage penalty, we have to maintain the coverage
        # of each prediction. We initialize it here to be the attention distribution
        # of the first token. It is ok not to penalize the first token because
        # the penalty should be 0.
        if self.coverage_penalizer is not None:
            # shape: (batch_size, beam_size, num_document_tokens)
            coverage = state['attention'].clone().view(batch_size, self.beam_size, -1)

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = self.predictions[-1].reshape(batch_size * self.beam_size)

            # If the last token was not the end index, we increment the length by 1
            self.lengths += (last_predictions != self._end_index).long()

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if self.is_finished():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            class_log_probabilities, state = step(last_predictions, state)

            # If the predictions have not reached the minimum length, prevent
            # the end index from being generated. The minimum number of steps
            # doesn't include the start or end index.
            if self.min_steps is not None and len(self.predictions) < self.min_steps:
                class_log_probabilities[:, self._end_index] = float('-inf')

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size,
                num_classes
            )

            # If there are ngrams which are disallowed, we prevent any token
            # which would repeat an ngram from being generated
            class_log_probabilities = class_log_probabilities.view(batch_size, self.beam_size, -1)
            self._mask_disallowed_ngrams(class_log_probabilities)
            class_log_probabilities = class_log_probabilities.view(batch_size * self.beam_size, -1)

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities
            )

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probabilities, predicted_classes = \
                cleaned_log_probabilities.topk(self.per_node_beam_size)

            if self.coverage_penalizer is not None:
                # To update the coverage penalty for this step, we first update
                # the coverage vector with the attention scores for this step,
                # compute the corresponding penalty, then add the peanlty to
                # the token log probabilites
                # shape: (batch_size, beam_size, num_document_tokens)
                coverage += state['attention'].view(batch_size, self.beam_size, -1)
                # shape: (batch_size * beam_size, per_node_beam_size)
                coverage_penalty = self.coverage_penalizer(coverage).\
                    unsqueeze(2).\
                    expand(batch_size, self.beam_size, self.per_node_beam_size).\
                    reshape(batch_size * self.beam_size, self.per_node_beam_size)
                # shape: (batch_size * beam_size, per_node_beam_size)
                top_log_probabilities += coverage_penalty

            # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probabilities = last_log_probabilities.\
                unsqueeze(2).\
                expand(batch_size, self.beam_size, self.per_node_beam_size).\
                reshape(batch_size * self.beam_size, self.per_node_beam_size)

            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities.\
                reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.\
                reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)

            self.predictions.append(restricted_predicted_classes)

            self.log_probs.append(restricted_beam_log_probs)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # Remove the coverage penalty to recover the true log-probabilities and
            # update the coverage vectors based on the selected indices
            if self.coverage_penalizer is not None:
                # shape: (batch_size, beam_size * per_node_beam_size)
                coverage_penalty = coverage_penalty.reshape(batch_size, self.beam_size * self.per_node_beam_size)
                # shape: (batch_size, beam_size)
                selected_coverage_penalties = coverage_penalty.gather(1, restricted_beam_indices)
                last_log_probabilities = last_log_probabilities - selected_coverage_penalties
                # shape: (batch_size, beam_size * per_node_beam_size, num_document_tokens)
                coverage = coverage.unsqueeze(2).\
                    expand(-1, -1, self.per_node_beam_size, -1).\
                    reshape(batch_size, self.beam_size * self.per_node_beam_size, -1)
                # shape: (batch_size, beam_size, num_document_tokens)
                reshaped_indices = restricted_beam_indices.unsqueeze(2).\
                    expand(-1, -1, coverage.size(-1))
                # shape: (batch_size, beam_size, num_document_tokens)
                coverage = coverage.gather(1, reshaped_indices)

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices / self.per_node_beam_size

            self.backpointers.append(backpointer)

            self.update_state()

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.\
                    view(batch_size, self.beam_size, *([1] * len(last_dims))).\
                    expand(batch_size, self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                state[key] = state_tensor.\
                    reshape(batch_size, self.beam_size, *last_dims).\
                    gather(1, expanded_backpointer).\
                    reshape(batch_size * self.beam_size, *last_dims)

        # Update the length one last time if the last token was not the end, then reshape
        self.lengths += (self.predictions[-1].reshape(batch_size * self.beam_size) != self._end_index).long()
        # shape: (batch_size, beam_size)
        self.lengths = self.lengths.view(batch_size, self.beam_size)

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces or the search was unable "
                          "to find a sequence longer than the minimum number of steps.",
                          RuntimeWarning)

        # Reconstruct the sequences using the backpointers
        final_predictions, final_log_probs, final_lengths = self.get_final_predictions()

        # Use the length penalizer to rerank the predictions if one is provided
        self._apply_length_penalty(final_predictions, final_log_probs, final_lengths)

        return final_predictions, final_log_probs

    def reuse_beam(self) -> bool:
        return False

    def update_state(self) -> None:
        pass

    def is_finished(self) -> bool:
        return (self.predictions[-1] == self._end_index).all()

    def get_final_predictions(self) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
        predictions = self._reconstruct_predictions(self.predictions, self.backpointers)
        return predictions, self.log_probs[-1], self.lengths

    @classmethod
    def from_params(cls, params: Params, vocab: Vocabulary) -> 'BeamSearch':
        type_ = params.pop('type', None)
        if type_ is not None:
            return cls.by_name(type_).from_params(params=params, vocab=vocab)

        beam_size = params.pop('beam_size')
        namespace = params.pop('namespace', 'tokens')
        end_symbol = params.pop('end_symbol', None)
        min_steps = params.pop('min_steps', None)
        max_steps = params.pop('max_steps', 50)
        per_node_beam_size = params.pop('per_node_beam_size', None)
        disallow_repeated_ngrams = params.pop('disallow_repeated_ngrams', None)
        repeated_ngrams_exceptions = params.pop('repeated_ngrams_exceptions', None)

        length_penalizer = None
        length_penalizer_params = params.pop('length_penalizer', None)
        if length_penalizer_params is not None:
            length_penalizer = LengthPenalizer.from_params(length_penalizer_params)

        coverage_penalizer = None
        coverage_penalizer_params = params.pop('coverage_penalizer', None)
        if coverage_penalizer_params is not None:
            coverage_penalizer = CoveragePenalizer.from_params(coverage_penalizer_params)

        return cls(vocab=vocab,
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


BeamSearch.register('standard')(BeamSearch)
