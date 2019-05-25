import torch
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, Vocabulary
from allennlp.training.metrics import Metric
from overrides import overrides
from typing import Dict, List, Optional, Union

from summarize.common.util import SENT_START_SYMBOL, SENT_END_SYMBOL
from summarize.metrics.python_rouge import PythonRouge


@Metric.register('python-rouge')
class PythonRougeMetric(Metric):
    """
    The ``PythonRougeMetric`` is an implementation of ROUGE that can be used
    while training models. The metrics computed over several different batches
    should be identical to calling the ``PythonRouge`` class once for the
    entire set of summaries.

    Parameters
    ----------
    ngram_orders: ``Union[int, List[int]]``
        The n-gram orders that should be computed. This should be the minimum
        number required for the fastest computation.
    max_sentences: ``int``, optional (default = ``None``)
        The maximum number of sentences to use for ROUGE. If ``None``, all are used.
    max_words: ``int``, optional (default = ``None``)
        The maximum number of words to use for ROUGE. If ``None``, all are used.
    max_bytes: ``int``, optional (default = ``None``)
        The maximum number of bytes to use for ROUGE. If ``None``, all are used.
    use_porter_stemmer: ``bool``, optional (default = ``True``)
        Indicates if the Porter Stemmer should be used.
    remove_stopwords: ``bool``, optional (default = ``False``)
        Indicates if stopwords should be removed from the summaries.
    namespace: ``str``, optional (default = ``"tokens"``)
        The summary token namespace.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 ngram_orders: Union[int, List[int]],
                 max_sentences: Optional[int] = None,
                 max_words: Optional[int] = None,
                 max_bytes: Optional[int] = None,
                 use_porter_stemmer: bool = True,
                 remove_stopwords: bool = False,
                 namespace: str = 'tokens') -> None:
        super().__init__()
        if isinstance(ngram_orders, int):
            ngram_orders = [ngram_orders]
        self.ngram_orders = ngram_orders
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.max_bytes = max_bytes
        self.use_porter_stemmer = use_porter_stemmer
        self.remove_stopwords = remove_stopwords
        self.python_rouge = PythonRouge()

        self.vocab = vocab
        self.namespace = namespace
        vocab_tokens = vocab.get_token_to_index_vocabulary(namespace)

        # Extract the special tokens from the vocabulary. We need to check and
        # ensure each one exists, otherwise we would get the OOV symbol, which
        # we don't want to skip when converting from indices to strings.
        self.start_index = None
        if START_SYMBOL in vocab_tokens:
            self.start_index = vocab_tokens[START_SYMBOL]
        self.end_index = None
        if END_SYMBOL in vocab_tokens:
            self.end_index = vocab_tokens[END_SYMBOL]
        self.pad_index = None
        if DEFAULT_PADDING_TOKEN in vocab_tokens:
            self.pad_index = vocab_tokens[DEFAULT_PADDING_TOKEN]
        self.sent_start_index = None
        if SENT_START_SYMBOL in vocab_tokens:
            self.sent_start_index = vocab_tokens[SENT_START_SYMBOL]
        self.sent_end_index = None
        if SENT_END_SYMBOL in vocab_tokens:
            self.sent_end_index = vocab_tokens[SENT_END_SYMBOL]

        self.count = 0
        self.totals = {}

    def _get_string_from_tensor(self, tensor: torch.Tensor) -> str:
        assert tensor.dim() == 1
        tokens = []
        for index in tensor:
            index = index.item()
            # We skip the start, sentence start, and sentence end symbols. It is
            # ok if these symbols are ``None`` since they won't match the index
            if index in [self.start_index, self.sent_start_index, self.sent_end_index]:
                continue
            # We end if we see the end or padding index
            if index in [self.end_index, self.pad_index]:
                break
            tokens.append(self.vocab.get_token_from_index(index, self.namespace))
        return tokens

    def _convert_to_strings(self, summaries: Union[List[str], List[List[str]], List[torch.Tensor], torch.Tensor]) -> List[List[str]]:
        """
        Converts the summaries into ``List[List[str]]``, where each individual
        summary is a ``List[str]``. Abstractive summaries will be a list of length 1.
        """
        # If the inner-most element is a string, these are already ok and they
        # just might need to be added to a new dimension (for abstractive). Otherwise,
        # we need to convert them from tensors into strings
        if isinstance(summaries, list):
            if isinstance(summaries[0], list):
                if isinstance(summaries[0][0], str):
                    # Extractive strings
                    return summaries
            elif isinstance(summaries[0], str):
                # Abstractive strings
                return [[summary] for summary in summaries]

        # Otherwise, this is a tensor
        return self._convert_tensor_to_strings(summaries)

    def _convert_tensor_to_strings(self, summaries: Union[List[torch.Tensor], torch.Tensor]) -> List[List[str]]:
        """
        Converts the summaries represented as tensors to ``List[List[str]]``, where
        each summary is a ``List[str]``.
        """
        # If the summaries have 2 dimensions, they are assumed to be (batch_size, num_tokens)
        # objects that has the tokens as a single sequence. This is generally done for
        # abstractive summarization. If the summaries have 3 dimensions, they are
        # assumed to be of size (batch_size, num_sents, num_tokens), which is common
        # with extractive summarization.
        summaries_strings = []
        for summary in summaries:
            if summary.dim() == 1:
                # Abstractive
                tokens = self._get_string_from_tensor(summary)
                if not tokens:
                    raise Exception(f'Summary has no tokens: {summary}')
                string = ' '.join(tokens)
                summaries_strings.append([string])
            elif summary.dim() == 2:
                # Extractive
                sentence_strings = []
                for sentence in summary:
                    tokens = self._get_string_from_tensor(sentence)
                    if tokens:
                        string = ' '.join(tokens)
                        sentence_strings.append(string)
                if not sentence_strings:
                    raise Exception(f'Summary has no tokens: {summary}')
                summaries_strings.append(sentence_strings)
            else:
                raise Exception(f'Summaries must be 1- or 2-dimensional')
        return summaries_strings

    @overrides
    def __call__(self,
                 gold_summaries: Union[List[str], List[List[str]], List[torch.Tensor], torch.Tensor],
                 model_summaries: Union[List[str], List[List[str]], List[torch.Tensor], torch.Tensor],
                 **kwargs) -> None:
        """
        Computes ROUGE based on the batched input summaries. The summaries can be represented
        with several different types, depending on the use case.

            - ``List[str]``: Each summary is a ``str``, used with abstractive summaries
            - ``List[List[str]]``: Each summary is a ``List[str]``, used with extractive summaries
            - ``List[torch.Tensor]``: Each summary is a ``torch.Tensor`` that is either 1- or 2-dimensional
                    for abstractive or extractive summaries, respectively
            - ``torch.Tensor``: Each summary is a ``torch.Tensor``. If the input is 2-dimensional, each
                    row is assumed to be a 1-dimensional abstractive summary. If the input is 3-dimensional,
                    each matrix is assumed to be a 2-dimensional extractive summary.

        Parameters
        ----------
        gold_summaries: ``Union[List[str], List[List[str]], List[torch.Tensor], torch.Tensor]``
            See above.
        model_summaries: ``Union[List[str], List[List[str]], List[torch.Tensor], torch.Tensor]``
            See above.
        """
        gold_summaries = self._convert_to_strings(gold_summaries)
        model_summaries = self._convert_to_strings(model_summaries)
        metrics = self.python_rouge.run_python_rouge(gold_summaries, model_summaries,
                                                     ngram_orders=self.ngram_orders,
                                                     max_sentences=self.max_sentences,
                                                     max_words=self.max_words,
                                                     max_bytes=self.max_bytes,
                                                     use_porter_stemmer=self.use_porter_stemmer,
                                                     remove_stopwords=self.remove_stopwords)

        num_instances = len(gold_summaries)
        self.count += num_instances
        for metric, value in metrics.items():
            if metric not in self.totals:
                self.totals[metric] = value * num_instances
            else:
                self.totals[metric] += value * num_instances

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric, value in self.totals.items():
            metrics[metric] = value / self.count
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self) -> None:
        self.count = 0
        self.totals.clear()
