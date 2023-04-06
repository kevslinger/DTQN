from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import dtqn.networks.drqn as drqn
from utils import torch_utils


class SoftAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        self.W = nn.Linear(embed_size, embed_size, bias=False)
        self.linear = nn.Linear(embed_size, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)

    def forward(self, x, h):
        # g(v_t^i, h_{t-1}) = softmax(Linear(Tanh(Linear(v_t^i) + Wh_{t-1})))
        y = self.W(h.transpose(1, 0))
        x = self.linear(x)
        z = x + y
        z = torch.tanh(z)
        z = self.linear2(z)
        return F.softmax(z, dim=2)


class DARQN(drqn.DRQN):
    """DARQN https://arxiv.org/pdf/1512.01693.pdf"""

    def __init__(
        self,
        input_shape: int,
        n_actions: int,
        embed_per_obs_dim: int,
        inner_embed: int,
        is_discrete_env: bool,
        obs_vocab_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            num_actions=n_actions,
            embed_per_obs_dim=embed_per_obs_dim,
            inner_embed=inner_embed,
            is_discrete_env=is_discrete_env,
            obs_vocab_size=obs_vocab_size,
            **kwargs,
        )
        self.hidden_zeros = nn.Parameter(
            torch.zeros(1, batch_size, inner_embed, dtype=torch.float32),
            requires_grad=False,
        )
        self.attention = SoftAttention(embed_size=inner_embed)
        self.apply(torch_utils.init_weights)

    def forward(
        self,
        x: torch.tensor,
        hidden_states: Optional[tuple] = None,
        episode_lengths: Optional[int] = None,
        padding_value: Optional[int] = None,
    ):
        x = self.obs_embed(x)
        # We only supply hidden states within an episode
        # If we don't supply hidden states, then we're doing a batch
        # forward pass
        if hidden_states is not None:
            attention = self.attention(x, hidden_states[0])
            lstm_out, hidden_states = self.lstm(attention, hidden_states)
            q_values = self.ffn(lstm_out)
        else:
            q_values = []
            hidden_states = (
                torch.zeros_like(self.hidden_zeros),
                torch.zeros_like(self.hidden_zeros),
            )
            context_len = x.size(1)
            for i in range(context_len):
                attention = self.attention(x[:, i : i + 1, :], hidden_states[0])
                lstm_out, hidden_states = self.lstm(attention, hidden_states)
                q = self.ffn(lstm_out)
                q_values.append(q)
            q_values = torch.cat(q_values, dim=1)

        return q_values, hidden_states
