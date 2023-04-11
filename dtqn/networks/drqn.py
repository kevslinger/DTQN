import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
from typing import Optional, Tuple
from dtqn.networks.dqn import DQN
from utils import torch_utils


class DRQN(DQN):
    """DRQN https://arxiv.org/pdf/1507.06527.pdf"""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int,
        action_dim: int,
        inner_embed: int,
        is_discrete_env: bool,
        obs_vocab_size: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            num_actions=num_actions,
            embed_per_obs_dim=embed_per_obs_dim,
            action_dim=action_dim,
            inner_embed_size=inner_embed,
            is_discrete_env=is_discrete_env,
            obs_vocab_size=obs_vocab_size,
            **kwargs
        )
        self.lstm = nn.LSTM(
            input_size=inner_embed, hidden_size=inner_embed, batch_first=True
        )
        self.apply(torch_utils.init_weights)

    def forward(
        self,
        obss: torch.Tensor,
        _: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor]] = None,
        episode_lengths: Optional[Tuple[int]] = None,
    ):
        token_embed = self.obs_embed(obss)
        # Hidden states are supplied when we are passing one observation
        # at a time (auto-regressively).
        if hidden_states is not None:
            lstm_out, new_hidden = self.lstm(token_embed, hidden_states)
        else:
            context_length = token_embed.size(1)
            packed_sequence = rnn.pack_padded_sequence(
                token_embed,
                episode_lengths.squeeze(),
                enforce_sorted=False,
                batch_first=True,
            )
            packed_output, new_hidden = self.lstm(packed_sequence)
            lstm_out, _ = rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=context_length,
            )

        q_values = self.ffn(lstm_out)
        return q_values, new_hidden
