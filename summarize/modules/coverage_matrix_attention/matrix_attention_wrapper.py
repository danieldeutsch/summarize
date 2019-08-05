import torch
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn.util import masked_softmax
from overrides import overrides
from typing import Tuple

from summarize.modules.coverage_matrix_attention import CoverageMatrixAttention


@CoverageMatrixAttention.register('matrix-attention')
class MatrixAttentionWrapper(CoverageMatrixAttention):
    """
    Wraps the ``MatrixAttention`` module from AllenNLP so the attention functions
    which do not use coverage can implement the ``CoverageMatrixAttention`` module
    interface.

    Parameters
    ----------
    matrix_attention: ``MatrixAttention``
        The underlying ``MatrixAttention`` to use.
    """
    def __init__(self, matrix_attention: MatrixAttention) -> None:
        super().__init__()
        self.matrix_attention = matrix_attention

    @overrides
    def forward(self,
                decoder_outputs: torch.Tensor,
                encoder_outputs: torch.Tensor,
                encoder_mask: torch.Tensor,
                coverage_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: (batch_size, num_summary_tokens, num_document_tokens)
        affinities = self.matrix_attention(decoder_outputs, encoder_outputs)
        # shape: (batch_size, num_summary_tokens, num_document_tokens)
        probabilities = masked_softmax(affinities, encoder_mask)

        # Create dummy coverage vectors to return
        batch_size, num_summary_tokens, num_document_tokens = affinities.size()
        # shape: (batch_size, num_summary_tokens, num_document_tokens)
        coverage_vectors = coverage_vector.new_zeros(batch_size, num_summary_tokens, num_document_tokens)
        # shape: (batch_size, num_document_tokens)
        coverage_vector = coverage_vector.new_zeros(batch_size, num_document_tokens)

        return probabilities, coverage_vectors, coverage_vector
