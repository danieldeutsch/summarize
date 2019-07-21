import torch
import unittest

from summarize.modules.coverage_matrix_attention import MLPCoverageAttention


class TestMLPCoverageAttention(unittest.TestCase):
    def test_mlp_coverage_attention(self):
        batch_size = 3
        num_encoder_tokens = 7
        num_decoder_tokens = 9
        encoder_dim = 11
        decoder_dim = 13
        attention_dim = 17

        encoder_outputs = torch.rand(batch_size, num_encoder_tokens, encoder_dim)
        encoder_mask = torch.LongTensor([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0]
        ])
        decoder_outputs = torch.rand(batch_size, num_decoder_tokens, decoder_dim)
        initial_coverage_vector = torch.zeros(batch_size, num_encoder_tokens)

        attention = MLPCoverageAttention(encoder_dim, decoder_dim, attention_dim)
        probabilities, coverage_vectors, coverage_vector = \
            attention(decoder_outputs, encoder_outputs, encoder_mask, initial_coverage_vector)

        # It's too hard to test specific values, so we run several sanity checks
        assert probabilities.size() == (batch_size, num_decoder_tokens, num_encoder_tokens)
        assert coverage_vectors.size() == (batch_size, num_decoder_tokens, num_encoder_tokens)
        assert coverage_vector.size() == (batch_size, num_encoder_tokens)

        # Make sure the first coverage vector is the initial argument
        assert torch.equal(initial_coverage_vector, coverage_vectors[:, 0])

        # Make sure the last coverage vector is the expected cumulative sum
        cumsum = torch.cumsum(probabilities, dim=1)
        assert torch.isclose(cumsum[:, -1], coverage_vector).all()

        # Make sure the probabilities obey the mask
        assert torch.equal((probabilities > 0).long(), encoder_mask.unsqueeze(1).expand_as(probabilities))
