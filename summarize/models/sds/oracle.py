from typing import List, Optional, Tuple, Union

from summarize.metrics.python_rouge import PythonRouge
from summarize.metrics.rouge import \
    R1_RECALL, R1_PRECISION, R1_F1, \
    R2_RECALL, R2_PRECISION, R2_F1, \
    R3_RECALL, R3_PRECISION, R3_F1, \
    R4_RECALL, R4_PRECISION, R4_F1, \
    RL_RECALL, RL_PRECISION, RL_F1


def get_greedy_oracle_summary(document: List[str],
                              summary: Union[List[str], List[List[str]]],
                              metric: str,
                              max_sentences: Optional[int] = None,
                              max_tokens: Optional[int] = None,
                              max_bytes: Optional[int] = None,
                              use_porter_stemmer: bool = True,
                              remove_stopwords: bool = False,
                              python_rouge: Optional[PythonRouge] = None) -> Tuple[List[str], List[int]]:
    """
    Computes the greedy oracle summary by selecting sentences from the
    input document while greedily increasing the metric until the summary budget
    is met. Exactly one of ``max_sentences``, ``max_tokens``, and ``max_bytes``
    must be not None.

    Parameters
    ----------
    document:
        The sentence-tokenized input document from which to extract the summary.
    summary:
        The ground-truth summary (``List[str]``) or summaries (``List[List[str]]``)
    metric:
        The name of the metric to greedily optimize. The metrics currently supported are
        the Rouge metrics (see `rouge.py`)
    max_sentences:
        The maximum number of allowed sentences to take.
    max_tokens:
        The maximum number of allowed tokens to take.
    max_bytes:
        The maximum number of allowed bytes to take.
    python_rouge:
        The PythonRouge object to use to compute the metrics, useful to avoid
        reloading the external resources on each call.

    Returns
    -------
    The summary (``List[str]``) and the corresponding sentence indices which
    were selected from the input document (``List[int]``).
    """
    if python_rouge is None:
        python_rouge = PythonRouge()

    if metric in [R1_RECALL, R1_PRECISION, R1_F1]:
        ngram_orders = [1]
        compute_rouge_l = False
    elif metric in [R2_RECALL, R2_PRECISION, R2_F1]:
        ngram_orders = [2]
        compute_rouge_l = False
    elif metric in [R3_RECALL, R3_PRECISION, R3_F1]:
        ngram_orders = [3]
        compute_rouge_l = False
    elif metric in [R4_RECALL, R4_PRECISION, R4_F1]:
        ngram_orders = [4]
        compute_rouge_l = False
    elif metric in [RL_RECALL, RL_PRECISION, RL_F1]:
        ngram_orders = []
        compute_rouge_l = True
    else:
        raise Exception(f'Unknown metric: {metric}')

    candidates = set(range(len(document)))
    selected = []
    current_score = None

    while len(candidates) > 0:
        max_index, max_score = None, None
        for index in candidates:
            candidate_summary = [document[index] for index in sorted(selected + [index])]
            metrics = python_rouge.run_python_rouge([summary], [candidate_summary],
                                                    ngram_orders=ngram_orders,
                                                    max_sentences=max_sentences,
                                                    max_tokens=max_tokens,
                                                    max_bytes=max_bytes,
                                                    use_porter_stemmer=use_porter_stemmer,
                                                    remove_stopwords=remove_stopwords,
                                                    compute_rouge_l=compute_rouge_l)
            score = metrics[metric]
            if max_score is None or score > max_score:
                max_score = score
                max_index = index

        if current_score is None or max_score > current_score:
            current_score = max_score
            selected.append(max_index)
            candidates.remove(max_index)
        else:
            break

    selected = list(sorted(selected))
    summary = [document[index] for index in selected]
    return summary, selected
