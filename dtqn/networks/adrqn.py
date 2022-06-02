import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from typing import Optional
from dtqn.networks.representations import EmbeddingRepresentation


class ADRQN(nn.Module):
    """ADRQN https://arxiv.org/pdf/1704.07978.pdf"""

    def __init__(
        self,
        input_shape: int,
        num_actions: int,
        embed_per_obs_dim: int,
        inner_embed_size: int,
        is_discrete_env: bool,
        obs_vocab_size: Optional[int] = None,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        super().__init__()
        if is_discrete_env:
            self.obs_embed = EmbeddingRepresentation.make_discrete_representation(
                vocab_sizes=obs_vocab_size,
                obs_dim=input_shape,
                embed_per_obs_dim=embed_per_obs_dim,
                outer_embed_size=inner_embed_size,
            )
        else:
            self.obs_embed = EmbeddingRepresentation.make_continuous_representation(
                obs_dim=input_shape, outer_embed_size=inner_embed_size
            )

        self.num_actions = num_actions
        action_embed_size = embed_per_obs_dim
        self.action_embed = nn.Embedding(num_actions, embed_per_obs_dim)
        self.zero_action = nn.Parameter(
            torch.zeros((batch_size, 1, action_embed_size)), requires_grad=False
        )

        self.total_embed_size = inner_embed_size + action_embed_size
        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.total_embed_size, inner_embed_size),
            nn.ReLU(),
            nn.Linear(inner_embed_size, num_actions),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
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
        x: torch.tensor,
        a: torch.tensor,
        hidden_states: Optional[tuple] = None,
        episode_lengths: Optional[int] = None,
        padding_value: Optional[int] = None,
        shift: Optional[bool] = False,
    ):
        x = self.obs_embed(x)
        a = self.action_embed(a)
        # We need to do a right shift of actions
        if shift:
            a = torch.cat((self.zero_action[: a.size(0)], a[:, :-1, :]), dim=1)
        # Concatenate the obs and action embeds before lstm
        x = torch.cat((x, a), dim=2)
        # Hidden states are supplied when we are passing one observation
        # at a time (auto-regressively).
        if hidden_states is not None:
            lstm_out, new_hidden = self.lstm(x, hidden_states)
        else:
            context_length = x.size(1)
            packed_sequence = rnn.pack_padded_sequence(
                x, episode_lengths, enforce_sorted=False, batch_first=True
            )
            packed_output, new_hidden = self.lstm(packed_sequence)
            lstm_out, _ = rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=context_length,
                padding_value=padding_value,
            )

        q_values = self.ffn(lstm_out)
        return q_values, new_hidden
