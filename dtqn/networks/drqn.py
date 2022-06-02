import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
from typing import Optional
from dtqn.networks.dqn import DQN


class DRQN(DQN):
    """DRQN https://arxiv.org/pdf/1507.06527.pdf"""

    def __init__(
        self,
        input_shape: int,
        num_actions: int,
        embed_per_obs_dim: int,
        inner_embed: int,
        is_discrete_env: bool,
        obs_vocab_size: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            num_actions=num_actions,
            embed_per_obs_dim=embed_per_obs_dim,
            inner_embed_size=inner_embed,
            is_discrete_env=is_discrete_env,
            obs_vocab_size=obs_vocab_size,
            **kwargs
        )
        self.lstm = nn.LSTM(
            input_size=inner_embed, hidden_size=inner_embed, batch_first=True
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
        hidden_states: Optional[tuple] = None,
        episode_lengths: Optional[int] = None,
        padding_value: Optional[int] = None,
    ):
        x = self.obs_embed(x)
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
