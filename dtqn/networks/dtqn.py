import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from dtqn.networks.representations import EmbeddingRepresentation
from dtqn.networks.gates import GRUGate, ResGate
from dtqn.networks.transformer import TransformerLayer, TransformerIdentityLayer


# This function taken from the torch transformer tutorial
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
def sinusoidal_pos(
    context_len: int,
    embed_dim: int,
):
    position = torch.arange(context_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    pos_encoding = torch.zeros(1, context_len, embed_dim)
    pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
    pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
    return pos_encoding


class DTQN(nn.Module):
    """Deep Transformer Q-Network for partially observable reinforcement learning.

    Args:
        obs_dim: The length of the observation vector.
        num_actions: The number of possible environments actions.
        embed_per_obs_dim: Used for discrete observation space. Length of the embed for each
            element in the observation dimension.
        inner_embed_size: The dimensionality of the network. Referred to as d_k by the
            original transformer.
        num_heads: The number of heads to use in the MultiHeadAttention.
        history_len: The maximum number of observations to take in.
        dropout: Dropout percentage. Default: `0.0`
        gate: Which layer to use after the attention and feedforward submodules (choices: `res`
            or `gru`). Default: `res`
        identity: Whether or not to use identity map reordering. Default: `False`
        pos: The kind of position encodings to use. `0` uses no position encodings, `1` uses
            learned position encodings, and `sin` uses sinusoidal encodings. Default: `1`
        discrete: Whether or not the environment has discrete observations. Default: `False`
        vocab_sizes: If discrete env only. Represents the number of observations in the
            environment. If the environment has multiple obs dims with different number
            of observations in each dim, this can be supplied as a vector. Default: `None`
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int,
        inner_embed_size: int,
        num_heads: int,
        num_layers: int,
        history_len: int,
        dropout: float = 0.0,
        gate: str = "res",
        identity: bool = False,
        pos: Union[str, int] = 1,
        discrete: bool = False,
        vocab_sizes: Optional[Union[np.ndarray, int]] = None,
        bag_size: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.discrete = discrete
        # Input Embedding
        if isinstance(obs_dim, tuple):
            self.obs_embedding = EmbeddingRepresentation.make_image_representation(
                obs_dim=obs_dim, outer_embed_size=inner_embed_size
            )
        else:
            if discrete:
                self.obs_embedding = (
                    EmbeddingRepresentation.make_discrete_representation(
                        vocab_sizes=vocab_sizes,
                        obs_dim=obs_dim,
                        embed_per_obs_dim=embed_per_obs_dim,
                        outer_embed_size=inner_embed_size,
                    )
                )
            else:
                self.obs_embedding = (
                    EmbeddingRepresentation.make_continuous_representation(
                        obs_dim=obs_dim, outer_embed_size=inner_embed_size
                    )
                )

        # If pos is 0, the positional embeddings are just 0s. Otherwise, they become learnables that initialise as 0s
        try:
            pos = int(pos)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, history_len, inner_embed_size),
                requires_grad=False if pos == 0 else True,
            )
        except ValueError:
            if pos == "sin":
                self.position_embedding = nn.Parameter(
                    sinusoidal_pos(context_len=history_len, embed_dim=inner_embed_size),
                    requires_grad=False,
                )
            else:
                raise AssertionError(f"pos must be either int or sin but was {pos}")

        self.bag_embed = nn.Parameter(
            torch.zeros(1, 1, inner_embed_size),
            requires_grad=True if bag_size > 0 else False,
        )
        self.dropout = nn.Dropout(dropout)

        if gate == "gru":
            attn_gate = GRUGate(embed_size=inner_embed_size)
            mlp_gate = GRUGate(embed_size=inner_embed_size)
        elif gate == "res":
            attn_gate = ResGate()
            mlp_gate = ResGate()
        if identity:
            transformer_block = TransformerIdentityLayer
        else:
            transformer_block = TransformerLayer
        self.transformer_layers = nn.Sequential(
            *[
                transformer_block(
                    num_heads,
                    inner_embed_size,
                    history_len + bag_size + 1,  # Need +1 for the bag eviction policy
                    dropout,
                    attn_gate,
                    mlp_gate,
                )
                for _ in range(num_layers)
            ]
        )

        self.layernorm = nn.LayerNorm(inner_embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(inner_embed_size, inner_embed_size),
            nn.ReLU(),
            nn.Linear(inner_embed_size, num_actions),
        )

        self.history_len = history_len
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.in_proj_bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, obss: torch.Tensor) -> torch.Tensor:
        history_len = obss.size(1)
        # if the observations are images, obs_dim is the dimensions of the image
        obs_dim = obss.size()[2:] if len(obss.size()) > 3 else obss.size(2)
        assert (
            history_len <= self.history_len
        ), "Cannot forward, history is longer than expected."
        assert (
            obs_dim == self.obs_dim
        ), f"Obs dim is incorrect. Expected {self.obs_dim} got {obs_dim}"

        token_embeddings = self.obs_embedding(obss)
        # batch_size x hist_len x obs_dim
        x = self.dropout(token_embeddings + self.position_embedding[:, :history_len, :])
        # Send through transformer
        x = self.transformer_layers(x)
        # Norm and run through a linear layer to get to action space
        x = self.layernorm(x)
        return self.ffn(x)

    def forward(
        self, obss: torch.Tensor, bag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, history_len, obs_dim = obss.size()
        assert (
            history_len <= self.history_len
        ), "Cannot forward, history is longer than expected."
        assert (
            obs_dim == self.obs_dim
        ), f"Obs dim is incorrect. Expected {self.obs_dim} got {obs_dim}"

        token_embeddings = self.obs_embedding(obss)
        # batch_size x hist_len x obs_dim
        x = token_embeddings + self.position_embedding[:, :history_len, :]
        if bag is not None:
            bag_embeddings = self.obs_embedding(bag)
            x = self.dropout(
                torch.cat(
                    (
                        x,
                        bag_embeddings + self.bag_embed,
                    ),
                    dim=1,
                )
            )
        else:
            x = self.dropout(x)
        # Send through transformer
        x = self.transformer_layers(x)
        # Norm and run through a linear layer to get to action space
        x = self.layernorm(x)
        return self.ffn(x)[:, :history_len, :]
