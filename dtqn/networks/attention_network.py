import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple
from dtqn.networks.representations import EmbeddingRepresentation


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


class ATTN(nn.Module):
    """Create a simple attention network, which has attention followed by an MLP.

    Args:
        num_heads: Number of heads to use for MultiHeadAttention.
        embed_size: The dimensionality of the layer.
        history_len: The maximum number of observations to take in.
        dropout: Dropout percentage.
    """

    def __init__(
        self,
        obs_dim: Union[int, Tuple],
        num_actions: int,
        embed_per_obs_dim: int,
        inner_embed_size: int,
        num_heads: int,
        history_len: int,
        dropout: float = 0.0,
        pos: Union[str, int] = "sin",
        discrete: bool = False,
        vocab_sizes: Optional[Union[np.ndarray, int]] = None,
    ):
        super().__init__()

        if discrete:
            self.obs_embedding = EmbeddingRepresentation.make_discrete_representation(
                vocab_sizes=vocab_sizes,
                obs_dim=obs_dim,
                embed_per_obs_dim=embed_per_obs_dim,
                outer_embed_size=inner_embed_size,
            )
        else:
            self.obs_embedding = EmbeddingRepresentation.make_continuous_representation(
                obs_dim=obs_dim, outer_embed_size=inner_embed_size
            )
        # If pos is 0, the positional embeddings are just 0s. Otherwise, they become learnables that initialise as 0s
        try:
            pos = int(pos)
            if pos == 0:
                self.register_buffer(
                    "position_embedding", torch.zeros(1, history_len, inner_embed_size)
                )
            else:
                self.position_embedding = nn.Parameter(
                    torch.zeros(1, history_len, inner_embed_size),
                )
        except ValueError:
            if pos == "sin":
                self.register_buffer(
                    "position_embedding",
                    sinusoidal_pos(context_len=history_len, embed_dim=inner_embed_size),
                )
                # self.position_embedding = nn.Parameter(
                #    sinusoidal_pos(context_len=history_len, embed_dim=inner_embed_size),
                #    requires_grad=False,
                # )
            else:
                raise AssertionError(f"pos must be either int or sin but was {pos}")
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.MultiheadAttention(
            embed_dim=inner_embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(inner_embed_size, 4 * inner_embed_size),
            nn.ReLU(),
            nn.Linear(4 * inner_embed_size, inner_embed_size),
        )
        self.to_actions = nn.Linear(inner_embed_size, num_actions)
        # Just storage for attention weights for visualization
        self.alpha = None

        # Set up causal masking for attention
        self.register_buffer(
            "attn_mask",
            torch.triu(
                torch.full((history_len, history_len), -float("inf")), diagonal=1
            ),
        )

        # The mask will look like:
        # [0, -inf, -inf, ..., -inf]
        # [0,    0, -inf, ..., -inf]
        # ...
        # [0,    0,    0, ...,    0]
        # Where 0 means that timestep is allowed to attend.
        # So the first timestep can attend only to the first timestep
        # and the last timestep can attend to all observations.

        self.history_len = history_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.MultiheadAttention)):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.in_proj_bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, history_len, obs_dim = x.size(0), x.size(1), x.size(2)
        assert (
            history_len <= self.history_len
        ), "Cannot forward, history is longer than expected."
        token_embeddings = self.obs_embedding(x)
        x = self.dropout(token_embeddings + self.position_embedding[:, :history_len, :])
        attention, self.alpha = self.attention(
            x,
            x,
            x,
            attn_mask=self.attn_mask[: x.size(1), : x.size(1)],
            average_attn_weights=True,  # Only affects self.alpha for visualizations
        )
        x = x + F.relu(attention)
        x = x + F.relu(self.ffn(x))
        return self.to_actions(x)
