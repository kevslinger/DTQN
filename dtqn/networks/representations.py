from __future__ import annotations
import torch.nn as nn
from typing import Optional



class EmbeddingRepresentation(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, obs):
        return self.embedding(obs)

    @staticmethod
    def make_discrete_representation(
        vocab_sizes: int, obs_dim: int, embed_per_obs_dim: int, outer_embed_size: int
    ) -> EmbeddingRepresentation:
        """
        For use in discrete observation environments.

        Args:
            vocab_sizes: The number of different values your observation could include.
            obs_dim: The length of the observation vector (assuming 1d).
            embed_per_obs_dim: The number of features you want to give to each observation
                dimension.
            embed_size: The length of the resulting embedding vector.
        """

        assert (
            vocab_sizes is not None
        ), "Discrete environments need to have a vocab size for the token embeddings"
        assert (
            embed_per_obs_dim > 1
        ), "Each observation feature needs at least 1 embed dim"

        embedding = nn.Sequential(
            nn.Embedding(vocab_sizes, embed_per_obs_dim),
            nn.Flatten(start_dim=-2),
            nn.Linear(embed_per_obs_dim * obs_dim, outer_embed_size),
        )
        return EmbeddingRepresentation(embedding=embedding)

    @staticmethod
    def make_continuous_representation(obs_dim: int, outer_embed_size: int):
        """
        For use in continuous observation environments. Projects the observation to the
            specified dimensionality for use in the network.

        Args:
            obs_dim: the length of the observation vector (assuming 1d)
            embed_size: The length of the resulting embedding vector
        """
        embedding = nn.Linear(obs_dim, outer_embed_size)
        return EmbeddingRepresentation(embedding=embedding)
