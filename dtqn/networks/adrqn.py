import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from typing import Optional, Tuple
from dtqn.networks.representations import (
    ObservationEmbeddingRepresentation,
    ActionEmbeddingRepresentation,
)


class ADRQN(nn.Module):
    """ADRQN https://arxiv.org/pdf/1704.07978.pdf"""

    def __init__(
        self,
        input_shape: int,
        num_actions: int,
        embed_per_obs_dim: int,
        action_dim: int,
        inner_embed_size: int,
        is_discrete_env: bool,
        obs_vocab_size: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__()
        if is_discrete_env:
            self.obs_embed = (
                ObservationEmbeddingRepresentation.make_discrete_representation(
                    vocab_sizes=obs_vocab_size,
                    obs_dim=input_shape,
                    embed_per_obs_dim=embed_per_obs_dim,
                    outer_embed_size=inner_embed_size - action_dim,
                )
            )
        else:
            self.obs_embed = (
                ObservationEmbeddingRepresentation.make_continuous_representation(
                    obs_dim=input_shape, outer_embed_size=inner_embed_size - action_dim
                )
            )

        self.num_actions = num_actions
        self.action_embed = ActionEmbeddingRepresentation(
            num_actions=num_actions, action_dim=action_dim
        )

        self.lstm = nn.LSTM(
            input_size=inner_embed_size,
            hidden_size=inner_embed_size,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(inner_embed_size, inner_embed_size),
            nn.ReLU(),
            nn.Linear(inner_embed_size, num_actions),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LSTM):
            module.weight_hh_l0.data.normal_(mean=0.0, std=0.02)
            module.weight_ih_l0.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias_hh_l0.data.zero_()
                module.bias_ih_l0.data.zero_()

    def forward(
        self,
        obss: torch.Tensor,
        actions: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor]] = None,
        episode_lengths: Optional[Tuple[int]] = None,
    ):
        history_len, context_length = obss.size(0), obss.size(1)
        obs_embed = self.obs_embed(obss)
        action_embed = self.action_embed(actions)
        # We need to do a right shift of actions
        if history_len > 1:
            action_embed = torch.roll(action_embed, 1, 1)
            # First observation in the sequence doesn't have a previous action, so zero the features
            action_embed[:, 0, :] = 0.0
        # Concatenate the obs and action embeds before lstm
        token_embed = torch.cat([action_embed, obs_embed], dim=-1)
        # Hidden states are supplied when we are passing one observation
        # at a time (auto-regressively).
        if hidden_states is not None:
            lstm_out, new_hidden = self.lstm(token_embed, hidden_states)
        else:
            packed_sequence = rnn.pack_padded_sequence(
                token_embed, episode_lengths, enforce_sorted=False, batch_first=True
            )
            packed_output, new_hidden = self.lstm(packed_sequence)
            lstm_out, _ = rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=context_length,
            )

        q_values = self.ffn(lstm_out)
        return q_values, new_hidden
