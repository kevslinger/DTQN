from __future__ import annotations
from typing import Tuple, Optional
import math

import torch
import torch.nn as nn


class ObservationEmbeddingRepresentation(nn.Module):
    def __init__(
        self,
        observation_embedding: nn.Module,
    ):
        super().__init__()
        self.observation_embedding = observation_embedding

    def forward(self, obs: torch.Tensor):
        batch, seq = obs.size(0), obs.size(1)
        # Flatten batch and seq dims
        obs = torch.flatten(obs, start_dim=0, end_dim=1)
        obs_embed = self.observation_embedding(obs)
        obs_embed = obs_embed.reshape(batch, seq, obs_embed.size(-1))
        return obs_embed

    @staticmethod
    def make_discrete_representation(
        vocab_sizes: int, obs_dim: int, embed_per_obs_dim: int, outer_embed_size: int
    ) -> ObservationEmbeddingRepresentation:
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
            vocab_sizes > 0
        ), "Discrete environments need to have a vocab size for the token embeddings"
        assert (
            embed_per_obs_dim > 1
        ), "Each observation feature needs at least 1 embed dim"

        embedding = nn.Sequential(
            nn.Embedding(vocab_sizes, embed_per_obs_dim),
            nn.Flatten(start_dim=-2),
            nn.Linear(embed_per_obs_dim * obs_dim, outer_embed_size),
        )
        return ObservationEmbeddingRepresentation(observation_embedding=embedding)

    @staticmethod
    def make_action_representation(
        num_actions: int,
        action_dim: int,
    ) -> ObservationEmbeddingRepresentation:
        embed = nn.Sequential(
            nn.Embedding(num_actions, action_dim), nn.Flatten(start_dim=-2)
        )
        return ObservationEmbeddingRepresentation(observation_embedding=embed)

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
        return ObservationEmbeddingRepresentation(observation_embedding=embedding)

    @staticmethod
    def make_image_representation(obs_dim: Tuple, outer_embed_size: int):
        """
        For use in image observatino environments.

        Args:
            obs_dim: The image observation's dimensions (C x H x W).
            outer_embed_size: The length of the resulting embedding vector.
        """
        # C x H x W or H x W
        if len(obs_dim) == 3:
            num_channels = obs_dim[0]
        else:
            num_channels = 1

        kernels = [3, 3, 3, 3, 3]
        paddings = [1, 1, 1, 1, 1]
        strides = [2, 1, 2, 1, 2]
        flattened_size = compute_flattened_size(
            obs_dim[1], obs_dim[2], kernels, paddings, strides
        )
        embedding = nn.Sequential(
            # Input 3 x 84 x 84
            nn.Conv2d(
                num_channels,
                64,
                kernel_size=kernels[0],
                padding=paddings[0],
                stride=strides[0],
            ),
            nn.ReLU(True),
            #
            nn.Conv2d(
                64, 64, kernel_size=kernels[1], padding=paddings[1], stride=strides[1]
            ),
            nn.ReLU(True),
            nn.Conv2d(
                64,
                64,
                kernel_size=kernels[2],
                padding=paddings[2],
                stride=strides[2],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                64, 128, kernel_size=kernels[3], padding=paddings[3], stride=strides[3]
            ),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * flattened_size, outer_embed_size),
        )
        return ObservationEmbeddingRepresentation(observation_embedding=embedding)


def compute_flattened_size(
    height: int, width: int, kernels: list, paddings: list, strides: list
) -> int:
    for i in range(len(kernels)):
        height = update_size(height, kernels[i], paddings[i], strides[i])
        width = update_size(width, kernels[i], paddings[i], strides[i])
    return int(height * width)


def update_size(component: int, kernel: int, padding: int, stride: int) -> int:
    return math.floor((component - kernel + 2 * padding) / stride) + 1


class ActionEmbeddingRepresentation(nn.Module):
    def __init__(self, num_actions: int, action_dim: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_actions, action_dim),
            nn.Flatten(start_dim=-2),
        )

    def forward(self, action: torch.Tensor):
        return self.embedding(action)
