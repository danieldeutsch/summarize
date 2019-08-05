import torch
from overrides import overrides

from summarize.modules.generate_probability_functions import GenerateProbabilityFunction


@GenerateProbabilityFunction.register('see2017')
class See2017GenerateProbabilityFunction(GenerateProbabilityFunction):
    """
    Computes the generation probability according to See et al. (2017). The probability
    is a linear function of the input embedding, output from the decoder (without attention),
    and the attention context vector.

    Parameters
    ----------
    embedding_dim: ``int``
        The size of the input embeddings to the decoder
    encoder_dim: ``int``
        The size of the encoder's hidden state.
    decoder_dim: ``int``
        The size of the decoder's hidden state.
    """
    def __init__(self, embedding_dim: int, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.input_layer = torch.nn.Linear(embedding_dim, 1)
        self.hidden_layer = torch.nn.Linear(decoder_dim, 1)
        self.context_layer = torch.nn.Linear(encoder_dim, 1)

    @overrides
    def forward(self,
                input_embeddings: torch.Tensor,
                pre_attention_decoder_outputs: torch.Tensor,
                post_attention_decoder_outputs: torch.Tensor,
                attention_context: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, num_summary_tokens)
        input_score = self.input_layer(input_embeddings).squeeze(2)
        # shape: (batch_size, num_summary_tokens)
        hidden_score = self.hidden_layer(pre_attention_decoder_outputs).squeeze(2)
        # shape: (batch_size, num_summary_tokens)
        context_score = self.context_layer(attention_context).squeeze(2)
        # shape: (batch_size, num_summary_tokens)
        probability = torch.sigmoid(context_score + hidden_score + input_score)

        # In my experience, the generation probability can sometimes be equal
        # to 1.0 or 0.0 (with really large/small scores) even with reasonably sized
        # parameter values. This causes problems with the log which is called
        # later on. Therefore, we move the probability closer to 0.5 by a small
        # number for stability.
        # shape: (batch_size, num_summary_tokens)
        geq_one_half_mask = (probability >= 0.5).float()
        # shape: (batch_size, num_summary_tokens)
        probability = (probability - 1e-3) * (geq_one_half_mask) + (probability + 1e-3) * (1 - geq_one_half_mask)
        return probability
