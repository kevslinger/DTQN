from typing import Optional, Tuple

import torch
import torch.nn as nn

from dtqn.networks.representations import (
    ObservationEmbeddingRepresentation,
    ActionEmbeddingRepresentation,
)
from dtqn.networks.position_encodings import PosEnum, PositionEncoding
from utils import torch_utils

"""
This file's code is largely adapted from the PerceiverIO repo
https://github.com/krasserm/perceiver-io/tree/737a76621852d0c618ef587341a957cc9cd9574f

"""


class QueryProvider:
    """Provider of cross-attention query input."""

    @property
    def num_query_channels(self):
        raise NotImplementedError()

    def __call__(self, x=None):
        raise NotImplementedError()


class TrainableQueryProvider(nn.Module, QueryProvider):
    """Provider of learnable cross-attention query input.

    This is the latent array in Perceiver IO encoders and the output query array in most Perceiver IO decoders.
    """

    def __init__(self, num_queries: int, query_dim: int, init_scale: float = 0.02):
        super().__init__()
        self._query = nn.Parameter(torch.empty(num_queries, query_dim))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self._query.normal_(0.0, init_scale)

    @property
    def num_query_channels(self):
        return self._query.shape[-1]

    def forward(self, x=None):
        return self._query.unsqueeze(0)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attention(x, x, x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xnorm = self.layernorm1(x)
        xQ, attn_weights = self.attention(xnorm)
        x = x + xQ
        return x + self.mlp(self.layernorm2(x))


class CrossAttention(nn.Module):
    def __init__(
        self, query_dim: int, keyvalue_dim: int, num_heads: int, dropout: float = 0.0
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=keyvalue_dim,
            vdim=keyvalue_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, xQ: torch.Tensor, xKV: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attention(xQ, xKV, xKV)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        query_embed_size: int,
        keyvalue_embed_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.xQlayernorm = nn.LayerNorm(query_embed_size)
        self.xKVlayernorm = nn.LayerNorm(keyvalue_embed_size)
        self.attention = CrossAttention(
            query_dim=query_embed_size,
            keyvalue_dim=keyvalue_embed_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(query_embed_size, query_embed_size),
            nn.ReLU(),
            nn.Linear(query_embed_size, query_embed_size),
        )
        self.layernorm2 = nn.LayerNorm(query_embed_size)

    def forward(
        self, xQ: torch.Tensor, xKV: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xQnorm = self.xQlayernorm(xQ)
        xKVnorm = self.xKVlayernorm(xKV)

        x, attn_weights = self.attention(xQnorm, xKVnorm)
        x = x + xQ

        return x + self.mlp(self.layernorm2(x))


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_array_size: int,
        num_latents: int,
        latent_array_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_provider = TrainableQueryProvider(num_latents, latent_array_size)
        self.cross_attn = CrossAttentionBlock(
            latent_array_size, input_array_size, num_heads, dropout
        )
        self.selfattn_layers = nn.Sequential(*[
            SelfAttentionBlock(latent_array_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_array = torch.concat([self.latent_provider() for _ in range(x.size(0))], dim=0)
        # latent_array = self.latent_provider().repeat(x.size(0), 1, 1)

        x_latent = self.cross_attn(latent_array, x)
        return self.selfattn_layers(x_latent)


class PerceiverDecoder(nn.Module):
    def __init__(
        self, num_queries: int, query_shape: int, keyvalue_shape: int, num_heads: int, dropout: float = 0.0
    ):
        super().__init__()
        self.output_query_provider = TrainableQueryProvider(num_queries, query_shape)
        self.cross_attn = CrossAttentionBlock(query_shape, keyvalue_shape, num_heads, dropout)

    def forward(self, x_latent: torch.Tensor) -> torch.Tensor:
        output_query_array = torch.concat([self.output_query_provider() for _ in range(x_latent.size(0))], dim=0)
        
        return self.cross_attn(output_query_array, x_latent)


class PerceiverIO(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int,
        action_dim: int,
        input_array_size: int,
        num_heads: int,
        num_layers: int,
        history_len: int,
        
        num_input_latents: int,
        input_latent_size: int,
        num_output_queries: int,
        pos: str,
        discrete: bool,
        vocab_sizes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Input Embedding: Allocate space for the action embedding
        obs_output_dim = input_array_size - action_dim
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
            context_len=history_len, embed_dim=input_array_size,
        )

        self.encoder = PerceiverEncoder(
            num_layers,
            input_array_size,
            num_input_latents,
            input_latent_size,
            num_heads,
            dropout,
        )
        self.decoder = PerceiverDecoder(
            num_output_queries, input_array_size, input_array_size, num_heads, dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(input_array_size, input_array_size),
            nn.ReLU(),
            nn.Linear(input_array_size, num_actions)
        )
        self.apply(torch_utils.init_weights)

    # TODO: kwargs
    def forward(self, x: torch.Tensor, actions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.obs_embedding(x)
        if self.action_embedding is not None:
            action_embed = self.action_embedding(actions)

            if x.size(1) > 1:
                action_embed = torch.roll(action_embed, 1, 1)
                action_embed[:, 0, :] = 0.0
            token_embeddings = torch.concat([action_embed, token_embeddings], dim=-1)
        t = token_embeddings + self.position_embedding()[:, :x.size(1), :]
        enc = self.encoder(t)
        dec = self.decoder(enc)
        return self.ffn(dec)
        #return self.ffn(self.decoder(self.encoder(token_embeddings + self.position_embedding()[:, :x.size(1), :])))
