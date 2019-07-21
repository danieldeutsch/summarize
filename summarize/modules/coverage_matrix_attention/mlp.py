import torch
from allennlp.nn.util import masked_softmax
from overrides import overrides
from typing import Tuple

from summarize.modules.coverage_matrix_attention import CoverageMatrixAttention


@CoverageMatrixAttention.register('mlp')
class MLPCoverageAttention(CoverageMatrixAttention):
    """
    An implementation of the coveraged-based MLP attention function from
    See et al. (2017).

    Parameters
    ----------
    encoder_size: ``int``
        The size of the encoder hidden states.
    decoder_size: ``int``
        The size of the decoder hidden states.
    attention_size: ``int``
        The size of the intermediate attention hidden size.
    """
    def __init__(self,
                 encoder_size: int,
                 decoder_size: int,
                 attention_size: int) -> None:
        super().__init__()
        self.linear_context = torch.nn.Linear(encoder_size, attention_size, bias=False)
        self.linear_query = torch.nn.Linear(decoder_size, attention_size, bias=True)
        self.v = torch.nn.Linear(attention_size, 1, bias=False)
        self.coverage_weights = torch.nn.Linear(1, attention_size, bias=False)

    @overrides
    def forward(self,
                decoder_outputs: torch.Tensor,
                encoder_outputs: torch.Tensor,
                encoder_mask: torch.Tensor,
                coverage_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We are able to precompute the encoder and decoder hidden projections
        # with vectorized computations instead of a for loop
        num_decoder_tokens = decoder_outputs.size(1)
        num_encoder_tokens = encoder_outputs.size(1)

        # shape: (batch_size, num_decoder_tokens, 1, decoder_size)
        decoder_outputs = decoder_outputs.unsqueeze(2)
        # shape: (batch_size, 1, num_encoder_tokens, encoder_size)
        encoder_outputs = encoder_outputs.unsqueeze(1)

        # shape: (batch_size, num_decoder_tokens, 1, attention_size)
        decoder_projection = self.linear_query(decoder_outputs)
        # shape: (batch_size, 1, num_encoder_tokens, attention_size)
        encoder_projection = self.linear_context(encoder_outputs)

        # shape: (batch_size, num_decoder_tokens, num_encoder_tokens, attention_size)
        decoder_projection = decoder_projection.expand(-1, -1, num_encoder_tokens, -1)
        # shape: (batch_size, num_decoder_tokens, num_encoder_tokens, attention_size)
        encoder_projection = encoder_projection.expand(-1, num_decoder_tokens, -1, -1)
        # shape: (batch_size, num_decoder_tokens, num_encoder_tokens, attention_size)
        joint_projection = decoder_projection + encoder_projection

        attention_probabilities = []
        coverage_vectors = []
        for i in range(num_decoder_tokens):
            # Save the vector that was used to compute this time step
            coverage_vectors.append(coverage_vector)

            # Compute the coverage-based term in the attention score
            # shape: (batch_size, num_encoder_tokens, 1)
            coverage_vector = coverage_vector.unsqueeze(2)
            # shape: (batch_size, num_encoder_tokens, attention_size)
            coverage_projection = self.coverage_weights(coverage_vector)

            # Compute the attention probabilities
            # shape: (batch_size, num_encoder_tokens)
            affinities = self.v(torch.tanh(joint_projection[:, i] + coverage_projection)).squeeze(2)
            # shape: (batch_size, num_encoder_tokens)
            probabilities = masked_softmax(affinities, encoder_mask)
            attention_probabilities.append(probabilities)

            # Update the coverage vector
            # shape: (batch_size, num_encoder_tokens)
            coverage_vector = coverage_vector.squeeze(2)
            # shape: (batch_size, num_encoder_tokens)
            coverage_vector = coverage_vector + probabilities

        # Prepare the tensors for output
        # shape: (batch_size, num_decoder_tokens, num_encoder_tokens)
        attention_probabilities = torch.stack(attention_probabilities, dim=1)
        # shape: (batch_size, num_decoder_tokens, num_encoder_tokens)
        coverage_vectors = torch.stack(coverage_vectors, dim=1)

        return attention_probabilities, coverage_vectors, coverage_vector
