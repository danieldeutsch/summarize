import torch
from allennlp.common.registrable import Registrable


class GenerateProbabilityFunction(torch.nn.Module, Registrable):
    def forward(self,
                input_embeddings: torch.Tensor,
                pre_attention_decoder_outputs: torch.Tensor,
                post_attention_decoder_outputs: torch.Tensor,
                attention_context: torch.Tensor) -> torch.Tensor:
        """
        Computes the probability of generating a token, the soft switch from
        See et al. (2017).

        Parameters
        ----------
        input_embeddings: (batch_size, num_summary_tokens, embedding_dim)
            The embeddings which are passed as input to the decoder.
        pre_attention_decoder_outputs: (batch_size, num_summary_tokens, hidden_dim)
            The direct output from the decoder, which does not include any attention.
        post_attention_decoder_outputs: (batch_size, num_summary_tokens, hidden_dim)
            The output of the decoder after attention has been included.
        attention_context: (batch_size, num_summary_tokens, encoder_hidden_dim)
            The attention context (the weighted average of the encoder hidden states
            based on the attention distribution)

        Returns
        -------
        (batch_size, num_summary_tokens):
            The generation probability.
        """
        raise NotImplementedError
