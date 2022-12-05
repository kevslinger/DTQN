import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F
from dtqn.buffers.replay_buffer import ReplayBuffer
import gym
from typing import Callable
import torch.optim as optim
from utils.logging_utils import RunningAverage

DEFAULT_CONTEXT_LEN = 50


class DqnAgent:
    def __init__(
            self,
            network_factory: Callable[[], Module],
            buffer_size: int,
            device: torch.device,
            env_obs_length: int,
            learning_rate: float = 0.0003,
            batch_size: int = 32,
            context_len: int = 50,
            gamma: float = 0.99,
            grad_norm_clip: float = 1.0,
            target_update_frequency: int = 10000,
            **kwargs,
    ):
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        # Initialize environment & networks
        self.policy_network = network_factory()
        self.target_network = network_factory()
        self.target_network.eval()
        # Ensure network's parameters are the same
        self.target_update()

        self.eval_on = self.policy_network.eval
        self.eval_off = self.policy_network.train

        optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env_obs_length=env_obs_length,
            context_len=context_len,
        )
        self.optimizer = optimizer
        self.device = device

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_frequency = target_update_frequency

        # Logging
        self.num_train_steps = 0
        self.num_evals = 0
        self.episode_rewards = RunningAverage(10)
        self.episode_lengths = RunningAverage(10)
        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

    @torch.no_grad()
    def get_action(self, current_obs) -> int:
        """Use policy_network to get an e-greedy action given the current obs.
        """
        # np.expand_dims(current_obs, axis=0)??
        q_values = self.policy_network(torch.as_tensor(current_obs, dtype=torch.float32, device=self.device))
        return torch.argmax(q_values).item()

    def store_transition(self, cur_obs, obs, action, reward, done, timestep):
        self.replay_buffer.store(cur_obs, obs, action, reward, done)

    def train(self) -> None:
        """Perform one gradient step of the network"""
        self.eval_off()
        if not self.replay_buffer.can_sample(self.batch_size):
            return
        obss, actions, rewards, next_obss, dones, _ = self.replay_buffer.sample(
            self.batch_size
        )

        # We pull obss/next_obss as [batch-size x 1 x obs-dim] so we need to squeeze it
        obss = torch.as_tensor(
            obss, dtype=torch.float32, device=self.device
        ).squeeze()
        next_obss = torch.as_tensor(
            next_obss, dtype=torch.float32, device=self.device
        ).squeeze()
        # Actions is [batch-size x 1 x 1] which we want to be [batch-size x 1]
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        ).squeeze(dim=1)
        # Rewards/Dones are [batch-size x 1 x 1] which we want to be [batch-size]
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).squeeze()
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device).squeeze()

        # obss is [batch-size x obs-dim] and after network is [batch-size x action-dim]
        # Then we gather it and squeeze to [batch-size]
        q_values = self.policy_network(obss).gather(1, actions).squeeze()

        with torch.no_grad():
            # We use DDQN, so the policy network determines which future actions we'd
            # take, but the target network determines the value of those
            next_obs_qs = self.policy_network(next_obss)
            argmax = torch.argmax(next_obs_qs, axis=1).unsqueeze(-1)
            next_obs_q_values = (
                self.target_network(next_obss).gather(1, argmax).squeeze()
            )

            # here goes BELLMAN
            targets = rewards + (1 - dones) * (next_obs_q_values * self.gamma)

        self.qvalue_max.add(q_values.max())
        self.qvalue_mean.add(q_values.mean())
        self.qvalue_min.add(q_values.min())

        self.target_max.add(targets.max())
        self.target_mean.add(targets.mean())
        self.target_min.add(targets.min())

        # Optimization step
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
        self.num_train_steps += 1

        if not self.num_train_steps % self.target_update_frequency:
            self.target_update()

    def target_update(self) -> None:
        """Hard update where we copy the network parameters from the policy network to the target network"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def context_reset(self):
        pass
