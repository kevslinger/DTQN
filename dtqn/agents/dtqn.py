from distutils.log import warn
from typing import Optional
from dtqn.agents.dqn import DqnAgent
from dtqn.agents.drqn import DrqnAgent
from utils import env_processing, epsilon_anneal
import numpy as np
import torch.nn.functional as F
import torch
import gym
from gym import spaces
import time
import random


class DtqnAgent(DrqnAgent):

    @torch.no_grad()
    def get_action(self, obs) -> int:
        # the policy network gets [1, timestep+1 x obs length] as input and
        # outputs [1, timestep+1 x 4 outputs]
        obs_history = self.context.get_history_of(obs)
        q_values = self.policy_network(
            torch.as_tensor(
                obs_history, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        )
        # We take the argmax of the last timestep's Q values
        # In other words, select the highest q value action
        return torch.argmax(q_values[:, -1, :]).item()

    def train(self) -> None:
        if not self.replay_buffer.can_sample(self.batch_size):
            return

        (
            obss,
            actions,
            rewards,
            next_obss,
            dones,
            episode_lengths,
        ) = self.replay_buffer.sample(self.batch_size)
        # Obss and Next obss: [batch-size x hist-len x obs-dim]
        obss = torch.as_tensor(obss, dtype=torch.float32, device=self.device)
        next_obss = torch.as_tensor(
            next_obss, dtype=torch.float32, device=self.device
        )
        # Actions: [batch-size x hist-len x 1]
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        # Rewards: [batch-size x hist-len x 1]
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        # Dones: [batch-size x hist-len x 1]
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)

        # obss is [batch-size x hist-len x obs-len]
        # then q_values is [batch-size x hist-len x n-actions]
        q_values = self.policy_network(obss)

        # After gathering, Q values becomes [batch-size x hist-len x 1] then
        # after squeeze becomes [batch-size x hist-len]
        if self.history:
            q_values = q_values.gather(2, actions).squeeze()
        else:
            q_values = q_values[:, -1, :].gather(1, actions[:, -1, :]).squeeze()

        with torch.no_grad():
            # Next obss goes from [batch-size x hist-len x obs-dim] to
            # [batch-size x hist-len x n-actions] and then goes through gather and squeeze
            # to become [batch-size x hist-len]
            if self.history:
                argmax = torch.argmax(self.policy_network(next_obss), dim=2).unsqueeze(
                    -1
                )
                next_obs_q_values = self.target_network(next_obss)
                next_obs_q_values = next_obs_q_values.gather(2, argmax).squeeze()
            else:
                argmax = torch.argmax(
                    self.policy_network(next_obss)[:, -1, :], dim=1
                ).unsqueeze(-1)
                next_obs_q_values = (
                    self.target_network(next_obss)[:, -1, :].gather(1, argmax).squeeze()
                )

            # here goes BELLMAN
            if self.history:
                targets = rewards.squeeze() + (1 - dones.squeeze()) * (
                        next_obs_q_values * self.gamma
                )
            else:
                targets = rewards[:, -1, :].squeeze() + (
                        1 - dones[:, -1, :].squeeze()
                ) * (next_obs_q_values * self.gamma)

        self.qvalue_max.add(q_values.max())
        self.qvalue_mean.add(q_values.mean())
        self.qvalue_min.add(q_values.min())
        self.target_max.add(targets.max())
        self.target_mean.add(targets.mean())
        self.target_min.add(targets.min())

        loss = F.mse_loss(q_values, targets)
        self.td_errors.add(loss)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.grad_norm_clip,
            error_if_nonfinite=True,
        )
        self.grad_norms.add(norm)

        self.optimizer.step()

