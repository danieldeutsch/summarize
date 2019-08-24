import argparse
import os
from collections import Counter, defaultdict
from tqdm import tqdm
from typing import Dict, List, Set

from summarize.data.io import JsonlReader, JsonlWriter


def _compute_probability_distribution(sentences: List[List[str]],
                                      beta: float) -> Dict[str, float]:
    # Compute the raw frequency counts for each token
    counts = Counter()
    for sentence in sentences:
        for token in sentence:
            counts[token] += 1

    # Add beta smoothing to each token
    for token in counts.keys():
        counts[token] += beta

    # Smooth into a probability distribution. We use a defaultdict with the
    # default value equal to observing an OOV token. Thus, the denominator needs
    # to include the smoothing for the OOV
    denominator = sum(counts.values()) + beta
    distribution = defaultdict(lambda: beta / denominator)
    for token, count in counts.items():
        distribution[token] = count / denominator
    return distribution


def _combine_probability_distributions(p_document: Dict[str, float],
                                       p_topic: Dict[str, float],
                                       p_context: Dict[str, float],
                                       document_lambda: float,
                                       topic_lambda: float,
                                       context_lambda: float) -> Dict[str, float]:
    # Computes the reweighted probability distribution over the tokens
    # in the *document only*
    p_combined = {}
    for token in p_document.keys():
        p_combined[token] = document_lambda * p_document[token] + topic_lambda * p_topic[token] + context_lambda * p_context[token]

    # We don't actually re-normalize this distribtuion. I believe that the
    # original SumBasic algorithm does not re-normalize the document distribution,
    # so we won't do it either.
    return p_combined


def _compute_sentence_weights(document: List[Set[str]],
                              candidates: Set[int],
                              p_combined: Dict[str, float]) -> Dict[int, float]:
    # Compute the weights for each of the candidate sentences
    weights = {}
    for index in candidates:
        sentence = document[index]
        weights[index] = sum(p_combined[token] for token in sentence) / len(sentence)
    return weights


def _get_highest_probability_tokens(probabilities: Dict[str, float]) -> str:
    tuples = list(probabilities.items())
    tuples.sort(key=lambda pair: -pair[1])
    return list(map(lambda pair: pair[0], tuples))


def run_sumfocus(document: List[str],
                 topics: List[str],
                 context: List[str],
                 beta: float,
                 topic_lambda: float,
                 context_lambda: float,
                 max_num_words: int = None,
                 max_num_sents: int = None) -> List[str]:
    if (max_num_words is not None) == (max_num_sents is not None):
        raise Exception('Exactly one of `max_num_words` and `max_num_sents` must be set.')

    assert 0 <= topic_lambda and topic_lambda <= 1
    assert 0 <= context_lambda and context_lambda <= 1
    assert topic_lambda + context_lambda <= 1
    document_lambda = 1.0 - topic_lambda - context_lambda

    # We assume the document and topic are tokenized by whitespace, so break
    # them into tokens and lowercase
    tokenized_document = [[token.lower() for token in sentence.split()] for sentence in document]
    tokenized_topic = [[token.lower() for token in sentence.split()] for sentence in topics]
    tokenized_context = [[token.lower() for token in sentence.split()] for sentence in context]

    # Step 1a: Compute the probability distribtuion over words.
    # We choose not to smooth the topic distribution because there are so few
    # tokens that smoothing on the same scale as the other two distributions would
    # change the signal
    p_document = _compute_probability_distribution(tokenized_document, beta)
    p_topic = _compute_probability_distribution(tokenized_topic, 0)
    p_context = _compute_probability_distribution(tokenized_context, beta)

    # Compute the length of each sentence in the document
    sentence_lengths = [len(sentence) for sentence in tokenized_document]

    # Convert the document into a set of words (tokens -> types conversion)
    document_types = [set(sentence) for sentence in tokenized_document]

    # Create a mapping from words to the list of sentences which contain those words
    token_to_sentence_indices = defaultdict(list)
    for i, sentence in enumerate(tokenized_document):
        for token in sentence:
            token_to_sentence_indices[token].append(i)

    candidates = set(range(len(tokenized_document)))
    selected = set()
    while len(candidates) > 0:
        # Step 1b: Compute the combined distribution
        p_combined = _combine_probability_distributions(p_document, p_topic, p_context,
                                                        document_lambda, topic_lambda, context_lambda)

        # Step 2: Compute the sentence weights
        weights = _compute_sentence_weights(document_types, candidates, p_combined)

        # Step 3: Pick the best scoring sentence that contains the highest probability token
        top_sentence = None
        best_tokens = _get_highest_probability_tokens(p_combined)
        for token in best_tokens:
            indices = token_to_sentence_indices[token]
            indices = [index for index in indices if index in candidates]
            if len(indices) == 0:
                # There are no more sentences with this token left. Move
                # on to the next one
                continue

            # Sort the candidate sentences by weight
            sentence_weights = [(index, weights[index]) for index in indices]
            sentence_weights.sort(key=lambda pair: -pair[1])

            # Take the best one
            top_sentence = sentence_weights[0][0]
            break

        # Update the selected sentences
        selected.add(top_sentence)
        candidates.remove(top_sentence)

        total_words = sum(sentence_lengths[index] for index in selected)
        if max_num_words is not None and total_words >= max_num_words:
            # Done computing the summary
            break
        if max_num_sents is not None and len(selected) >= max_num_sents:
            # Done computing the summary
            break

        # Step 4: Update the probabilities
        for token in p_document.keys():
            p_document[token] = p_document[token] ** 2

    summary = [document[index] for index in sorted(selected)]
    return summary


def main(args):
    dirname = os.path.dirname(args.output_jsonl)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in tqdm(f):
                document = instance['document']
                topics = instance['topics']
                context = instance['context']
                cloze = run_sumfocus(document, topics, context, args.beta, args.topic_lambda, args.context_lambda, args.max_words, args.max_sentences)
                cloze = ' '.join(cloze)
                out.write({'cloze': cloze})


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl')
    argp.add_argument('output_jsonl')
    argp.add_argument('beta', type=float, help='The token probability smoothing parameter')
    argp.add_argument('topic_lambda', type=float, help='The distribution interpolation parameter for the topic distribution')
    argp.add_argument('context_lambda', type=float, help='The distribution interpolation parameter for the context distribution')
    argp.add_argument('--max-words', type=int)
    argp.add_argument('--max-sentences', type=int)
    args = argp.parse_args()
    main(args)
