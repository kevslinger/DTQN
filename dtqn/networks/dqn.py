import torch.nn as nn
import torch
from typing import Optional
from dtqn.networks.representations import ObservationEmbeddingRepresentation


class DQN(nn.Module):
    """DQN https://www.nature.com/articles/nature14236.pdf"""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int,
        inner_embed_size: int,
        is_discrete_env: bool,
        obs_vocab_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        # Input Embedding
        if isinstance(obs_dim, tuple):
            self.obs_embed = (
                ObservationEmbeddingRepresentation.make_image_representation(
                    obs_dim=obs_dim, outer_embed_size=inner_embed_size
                )
            )
        else:
            if is_discrete_env:
                self.obs_embed = (
                    ObservationEmbeddingRepresentation.make_discrete_representation(
                        vocab_sizes=obs_vocab_size,
                        obs_dim=obs_dim,
                        embed_per_obs_dim=embed_per_obs_dim,
                        outer_embed_size=inner_embed_size,
                    )
                )
            else:
                self.obs_embed = (
                    ObservationEmbeddingRepresentation.make_continuous_representation(
                        obs_dim=obs_dim, outer_embed_size=inner_embed_size
                    )
                )

        self.ffn = nn.Sequential(
            nn.Linear(inner_embed_size, inner_embed_size),
            nn.ReLU(),
            nn.Linear(inner_embed_size, num_actions),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.tensor):
        return self.ffn(self.obs_embed(x))
