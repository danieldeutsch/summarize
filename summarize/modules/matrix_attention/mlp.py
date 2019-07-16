import torch
from allennlp.modules.matrix_attention import MatrixAttention
from overrides import overrides


@MatrixAttention.register('mlp')
class MLPAttention(MatrixAttention):
    """
    An implementation of the "concat" attention from the arvix version of
    Luong et al. (2015) (https://arxiv.org/pdf/1508.04025.pdf). For some reason,
    the "concat" attention is different in the version in the ACL Anthology.

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

    @overrides
    def forward(self,
                decoder_outputs: torch.Tensor,
                encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        decoder_outputs: ``torch.Tensor``, ``(batch_size, num_summary_tokens, decoder_size)``
            The decoder outputs
        encoder_outputs: ``torch.Tensor``, ``(batch_size, num_document_tokens, encoder_size)``

        Returns
        -------
        A ``(batch_size, num_summary_tokens, num_document_tokens)``-sized tensor with the
            unnormalized attention scores.
        """
        num_decoder_tokens = decoder_outputs.size(1)
        num_encoder_tokens = encoder_outputs.size(1)

        # shape: (batch_size, num_summary_tokens, 1, decoder_size)
        decoder_outputs = decoder_outputs.unsqueeze(2)
        # shape: (batch_size, 1, num_document_tokens, encoder_size)
        encoder_outputs = encoder_outputs.unsqueeze(1)

        # shape: (batch_size, num_summary_tokens, 1, attention_size)
        decoder_projection = self.linear_query(decoder_outputs)
        # shape: (batch_size, 1, num_document_tokens, attention_size)
        encoder_projection = self.linear_context(encoder_outputs)

        # shape: (batch_size, num_summary_tokens, num_document_tokens, attention_size)
        decoder_projection = decoder_projection.expand(-1, -1, num_encoder_tokens, -1)
        # shape: (batch_size, num_summary_tokens, num_document_tokens, attention_size)
        encoder_projection = encoder_projection.expand(-1, num_decoder_tokens, -1, -1)

        # shape: (batch_size, num_summary_tokens, num_document_tokens)
        affinities = self.v(torch.tanh(decoder_projection + encoder_projection)).squeeze(-1)
        return affinities
