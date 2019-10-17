import torch
from overrides import overrides

from summarize.modules.generate_probability_functions import GenerateProbabilityFunction


@GenerateProbabilityFunction.register('onmt')
class ONMTGenerateProbabilityFunction(GenerateProbabilityFunction):
    """
    Computes the generation probability according the function used by the
    OpenNMT framework. The probability is a function of only the final decoder
    hidden states (with attention).

    Parameters
    ----------
    decoder_dim: ``int``
        The size of the decoder's hidden state.
    """
    def __init__(self, decoder_dim: int) -> None:
        super().__init__()
        self.hidden_layer = torch.nn.Linear(decoder_dim, 1)

    @overrides
    def forward(self,
                input_embeddings: torch.Tensor,
                pre_attention_decoder_outputs: torch.Tensor,
                post_attention_decoder_outputs: torch.Tensor,
                attention_context: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, num_summary_tokens)
        return torch.sigmoid(self.hidden_layer(post_attention_decoder_outputs).squeeze(2))
