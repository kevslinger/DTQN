import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from dtqn.networks.representations import (
    ObservationEmbeddingRepresentation,
    ActionEmbeddingRepresentation,
)
from dtqn.networks.position_encodings import PosEnum, PositionEncoding
from dtqn.networks.gates import GRUGate, ResGate
from dtqn.networks.transformer import TransformerLayer, TransformerIdentityLayer
from utils import torch_utils


class DTQN(nn.Module):
    """Deep Transformer Q-Network for partially observable reinforcement learning.

    Args:
        obs_dim:            The length of the observation vector.
        num_actions:        The number of possible environments actions.
        embed_per_obs_dim:  Used for discrete observation space. Length of the embed for each
            element in the observation dimension.
        action_dim:         The number of features to give the action.
        inner_embed_size:   The dimensionality of the network. Referred to as d_k by the
            original transformer.
        num_heads:          The number of heads to use in the MultiHeadAttention.
        num_layers:         The number of transformer blocks to use.
        history_len:        The maximum number of observations to take in.
        dropout:            Dropout percentage. Default: `0.0`
        gate:               Which layer to use after the attention and feedforward submodules (choices: `res`
            or `gru`). Default: `res`
        identity:           Whether or not to use identity map reordering. Default: `False`
        pos:                The kind of position encodings to use. `0` uses no position encodings, `1` uses
            learned position encodings, and `sin` uses sinusoidal encodings. Default: `1`
        discrete:           Whether or not the environment has discrete observations. Default: `False`
        vocab_sizes:        If discrete env only. Represents the number of observations in the
            environment. If the environment has multiple obs dims with different number
            of observations in each dim, this can be supplied as a vector. Default: `None`
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int,
        action_dim: int,
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
        # Input Embedding: Allocate space for the action embedding
        obs_output_dim = inner_embed_size - action_dim
        if action_dim > 0:
            self.action_embedding = ActionEmbeddingRepresentation(
                num_actions=num_actions, action_dim=action_dim
            )
        else:
            self.action_embedding = None
        # Image observation domains
        if isinstance(obs_dim, tuple):
            self.obs_embedding = (
                ObservationEmbeddingRepresentation.make_image_representation(
                    obs_dim=obs_dim, outer_embed_size=obs_output_dim
                )
            )
        # Discrete or MultiDiscrete observation domains
        elif discrete:
            self.obs_embedding = (
                ObservationEmbeddingRepresentation.make_discrete_representation(
                    vocab_sizes=vocab_sizes,
                    obs_dim=obs_dim,
                    embed_per_obs_dim=embed_per_obs_dim,
                    outer_embed_size=obs_output_dim,
                )
            )
        # Continuous observation domains
        else:
            self.obs_embedding = (
                ObservationEmbeddingRepresentation.make_continuous_representation(
                    obs_dim=obs_dim, outer_embed_size=obs_output_dim
                )
            )

        pos_function_map = {
            PosEnum.LEARNED: PositionEncoding.make_learned_position_encoding,
            PosEnum.SIN: PositionEncoding.make_sinusoidal_position_encoding,
            PosEnum.NONE: PositionEncoding.make_empty_position_encoding,
        }
        self.position_embedding = pos_function_map[PosEnum(pos)](
            context_len=history_len, embed_dim=inner_embed_size
        )

        self.dropout = nn.Dropout(dropout)

        if gate == "gru":
            attn_gate = GRUGate(embed_size=inner_embed_size)
            mlp_gate = GRUGate(embed_size=inner_embed_size)
        elif gate == "res":
            attn_gate = ResGate()
            mlp_gate = ResGate()
        else:
            raise ValueError("Gate must be one of `gru`, `res`")

        if identity:
            transformer_block = TransformerIdentityLayer
        else:
            transformer_block = TransformerLayer
        self.transformer_layers = nn.Sequential(
            *[
                transformer_block(
                    num_heads,
                    inner_embed_size,
                    history_len,
                    dropout,
                    attn_gate,
                    mlp_gate,
                )
                for _ in range(num_layers)
            ]
        )

        self.bag_size = bag_size
        self.bag_attn_weights = None
        if bag_size > 0:
            self.bag_attention = nn.MultiheadAttention(
                inner_embed_size,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(inner_embed_size * 2, inner_embed_size),
                nn.ReLU(),
                nn.Linear(inner_embed_size, num_actions),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(inner_embed_size, inner_embed_size),
                nn.ReLU(),
                nn.Linear(inner_embed_size, num_actions),
            )

        self.history_len = history_len
        self.apply(torch_utils.init_weights)

    def forward(
        self,
        obss: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        bag_obss: Optional[torch.Tensor] = None,
        bag_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        obss    is  batch x seq_len  x obs_dim
        actions is  batch x seq_len  x       1
        bag     is  batch x bag_size x obs_dim
        """
        history_len = obss.size(1)
        assert (
            history_len <= self.history_len
        ), "Cannot forward, history is longer than expected."

        # If the observations are images, obs_dim is the dimensions of the image
        obs_dim = obss.size()[2:] if len(obss.size()) > 3 else obss.size(2)
        assert (
            obs_dim == self.obs_dim
        ), f"Obs dim is incorrect. Expected {self.obs_dim} got {obs_dim}"

        token_embeddings = self.obs_embedding(obss)

        # Just to keep shapes correct if we choose to disble including actions
        if self.action_embedding is not None:
            # [batch x seq_len x 1] -> [batch x seq_len x action_embed]
            action_embed = self.action_embedding(actions)

            if history_len > 1:
                action_embed = torch.roll(action_embed, 1, 1)
                # First observation in the sequence doesn't have a previous action, so zero the features
                action_embed[:, 0, :] = 0.0
            token_embeddings = torch.concat([action_embed, token_embeddings], dim=-1)

        # [batch x seq_len x model_embed] -> [batch x seq_len x model_embed]
        working_memory = self.transformer_layers(
            self.dropout(
                token_embeddings + self.position_embedding()[:, :history_len, :]
            )
        )

        if self.bag_size > 0:
            # [batch x bag_size x action_embed] + [batch x bag_size x obs_embed] -> [batch x bag_size x model_embed]
            if self.action_embedding is not None:
                bag_embeddings = torch.concat(
                    [self.action_embedding(bag_actions), self.obs_embedding(bag_obss)],
                    dim=-1,
                )
            else:
                bag_embeddings = self.obs_embedding(bag_obss)
            # [batch x seq_len x model_embed] x [batch x bag_size x model_embed] -> [batch x seq_len x model_embed]
            persistent_memory, self.attn_weights = self.bag_attention(
                working_memory, bag_embeddings, bag_embeddings
            )
            output = self.ffn(torch.concat([working_memory, persistent_memory], dim=-1))
        else:
            output = self.ffn(working_memory)

        return output[:, -history_len:, :]
