import numpy as np
from allennlp.data import Token
from allennlp.data.fields import ArrayField, ListField
from collections import defaultdict
from typing import Dict, List, Tuple


def get_token_to_index_map(tokens: List[Token]) -> Dict[str, List[int]]:
    """
    Creates a mapping from each token to the indices in ``tokens`` which it appears.
    The ``Token``s are converted to strings for processing.

    Parameters
    ----------
    tokens:
        The list of tokens to create an index mapping for.

    Returns
    -------
    The mapping from the string version of the token to the list of indices
    in ``tokens`` in which it appears.
    """
    token_to_indices = defaultdict(list)
    for i, token in enumerate(tokens):
        token_to_indices[str(token)].append(i)
    return token_to_indices


def get_first_indices_field(tokens: List[Token],
                            token_to_indices: Dict[str, List[int]]) -> Dict[str, int]:
    """
    Creates an ``ArrayField`` that contains the first index that each token
    of ``tokens`` appears in ``tokens`` according to ``token_to_indices``. The
    ``token_to_indices`` index lists should be sorted.

    Parameters
    ----------
    tokens:
        The list of tokens for which to create the field.
    token_to_indices:
        The mapping from the string version of the tokens to the indices that
        it appears, sorted.

    Returns
    -------
    An ``ArrayField`` with one entry for each token.
    """
    first_indices = []
    for token in tokens:
        first_index = token_to_indices[str(token)][0]
        first_indices.append(first_index)
    return ArrayField(np.array(first_indices))


def get_token_mapping_field(token_to_document_indices: Dict[str, List[int]],
                            summary: List[Token]) -> Tuple[ListField, ListField]:
    """
    Creates an ``ArrayField`` that, for each token in the summary, contains
    the list of document indices for which that token appears, plus the
    corresponding mask.

    Parameters
    ----------
    token_to_document_indices:
        The mapping from each token to the list of indices in the document it appears.
    summary:
        The summary tokens.

    Returns
    -------
    ``ArrayField``: (num_summary_tokens, max_num_matches)
        The mapping field.
    ``ArrayField``: (num_summary_tokens, max_num_matches)
        The corresponding mask.
    """
    summary_token_document_indices = []
    mask = []
    for token in summary:
        indices = token_to_document_indices[str(token)]
        summary_token_document_indices.append(ArrayField(np.array(indices)))
        mask.append(ArrayField(np.ones(len(indices))))

    # Convert these into fields
    summary_token_document_indices_field = ListField(summary_token_document_indices)
    mask_field = ListField(mask)
    return summary_token_document_indices_field, mask_field
