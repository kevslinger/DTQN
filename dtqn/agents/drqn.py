from typing import Callable, Union

import torch
from torch.nn import Module
import torch.nn.functional as F
import numpy as np

from dtqn.agents.dqn import DqnAgent
from utils.env_processing import Context


class DrqnAgent(DqnAgent):
    # noinspection PyTypeChecker
    def __init__(
        self,
        network_factory: Callable[[], Module],
        buffer_size: int,
        device: torch.device,
        env_obs_length: int,
        max_env_steps: int,
        obs_mask: Union[int, float],
        num_actions: int,
        is_discrete_env: bool,
        learning_rate: float = 0.0003,
        batch_size: int = 32,
        context_len: int = 50,
        gamma: float = 0.99,
        grad_norm_clip: float = 1.0,
        target_update_frequency: int = 10_000,
        embed_size: int = 64,
        history: bool = True,
        **kwargs,
    ):
        super().__init__(
            network_factory,
            buffer_size,
            device,
            env_obs_length,
            max_env_steps,
            obs_mask,
            num_actions,
            is_discrete_env,
            learning_rate,
            batch_size,
            context_len,
            gamma,
            grad_norm_clip,
            target_update_frequency,
            **kwargs,
        )
        self.history = history
        self.zeros_hidden = torch.zeros(
            1,
            1,
            embed_size,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        hidden_states = (self.zeros_hidden, self.zeros_hidden)

        self.train_context = Context(
            context_len,
            obs_mask,
            num_actions,
            env_obs_length,
            init_hidden=hidden_states,
        )
        self.eval_context = Context(
            context_len,
            obs_mask,
            num_actions,
            env_obs_length,
            init_hidden=hidden_states,
        )

    def observe(self, obs, action, reward, done) -> None:
        self.context.add_transition(obs, action, reward, done)
        if self.train_mode:
            o, a, r, d = self.context.export()
            self.replay_buffer.store(
                o,
                a,
                r,
                d,
                min(
                    self.context_len, self.context.timestep + 1
                ),  # TODO: Is +1 necessary?
            )

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, epsilon=0.0) -> int:
        q_values, self.context.hidden = self.policy_network(
            torch.as_tensor(
                obs,
                dtype=self.obs_tensor_type,
                device=self.device,
            ).reshape(1, 1, self.env_obs_length),
            hidden_states=self.context.hidden,
        )
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            return torch.argmax(q_values).item()

    def train(self) -> None:
        if not self.replay_buffer.can_sample(self.batch_size):
            return

        self.eval_off()
        (
            obss,
            actions,
            rewards,
            next_obss,
            dones,
            episode_lengths,
        ) = self.replay_buffer.sample(self.batch_size)

        # We pull obss/next_obss as [batch-size x context-len x obs-dim]
        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(
            next_obss, dtype=self.obs_tensor_type, device=self.device
        )
        # Actions starts as [batch-size x context-len x 1]
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        # Rewards/dones start as [batch-size x context-len x 1], squeeze to [batch-size x context-len]
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)
        episode_lengths = torch.as_tensor(
            episode_lengths, dtype=torch.long, device=torch.device("cpu")
        )

        # obss is [batch-size x context-len x obs-len]
        # then q_values is [batch-size x context-len x n-actions]
        q_values, _ = self.policy_network(
            obss, episode_lengths=episode_lengths, padding_value=self.obs_mask
        )

        if self.history:
            # After gathering, Q values becomes [batch-size x hist-len x 1], which we squeeze to [batch-size x hist-len]
            q_values = q_values.gather(2, actions).squeeze()
        else:
            # After gathering, Q values becomes [batch-size x 1], which we squeeze to [batch-size]
            q_values = q_values[:, -1, :].gather(1, actions[:, -1, :]).squeeze()

        with torch.no_grad():

            if self.history:
                # Next obss goes from [batch-size x context-len x obs-len] to
                # [batch-size x hist-len x n-actions] after forward pass, then indexed
                # to become [batch-size x hist-len x n-actions], then argmax/squeezed to become [batch-size x hist-len]
                argmax = torch.argmax(
                    self.policy_network(
                        next_obss,
                        episode_lengths=episode_lengths,
                        padding_value=self.obs_mask,
                    )[0],
                    dim=2,
                ).unsqueeze(-1)
                next_obs_q_values = (
                    self.target_network(
                        next_obss,
                        episode_lengths=episode_lengths,
                        padding_value=self.obs_mask,
                    )[0]
                    .gather(2, argmax)
                    .squeeze()
                )
                # here goes BELLMAN
                targets = rewards.squeeze() + (1 - dones.squeeze()) * (
                    next_obs_q_values * self.gamma
                )
            else:
                # Next obss goes from [batch-size x context-len x obs-len] to
                # [batch-size x hist-len x n-actions] after forward pass, then indexed
                # to become [batch-size x hist-len x 1]
                argmax = torch.argmax(
                    self.policy_network(
                        next_obss,
                        episode_lengths=episode_lengths,
                        padding_value=self.obs_mask,
                    )[0][:, -1, :],
                    dim=1,
                ).unsqueeze(-1)
                next_obs_q_values = (
                    self.target_network(
                        next_obss,
                        episode_lengths=episode_lengths,
                        padding_value=self.obs_mask,
                    )[0][:, -1, :]
                    .gather(1, argmax)
                    .squeeze()
                )

                # here goes BELLMAN
                targets = rewards[:, -1, :].squeeze() + (
                    1 - dones[:, -1, :].squeeze()
                ) * (next_obs_q_values * self.gamma)

        self.qvalue_max.add(q_values.max().item())
        self.qvalue_mean.add(q_values.mean().item())
        self.qvalue_min.add(q_values.min().item())

        self.target_max.add(targets.max().item())
        self.target_mean.add(targets.mean().item())
        self.target_min.add(targets.min().item())

        # Optimization step
        loss = F.mse_loss(q_values, targets)
        self.td_errors.add(loss.item())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.grad_norm_clip,
            error_if_nonfinite=True,
        )
        self.grad_norms.add(norm.item())
        self.optimizer.step()
        self.num_train_steps += 1

        if self.num_train_steps % self.target_update_frequency == 0:
            self.target_update()
