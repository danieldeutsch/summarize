import torch
from allennlp.common import Registrable


class SentenceExtractor(torch.nn.Module, Registrable):
    def forward(self,
                sentence_encodings: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the probability of each sentence being extracted from the
        sentence encodings.

        Parameters
        ----------
        sentence_encodings: (batch_size, num_sents, hidden_dim)
            The encoding of each sentence
        mask: (batch_size, num_sents)
            The sentence mask

        Returns
        -------
        A (batch_size, num_sents) tensor with the raw extraction scores for each
        input sentence.
        """
        raise NotImplementedError
