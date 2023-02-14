from typing import Callable, Union, Tuple
import random

import torch
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import joblib

from dtqn.buffers.replay_buffer import ReplayBuffer
from utils.logging_utils import RunningAverage
from utils.env_processing import Context
from utils.epsilon_anneal import LinearAnneal


class DqnAgent:
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
        context_len: int = 1,
        gamma: float = 0.99,
        grad_norm_clip: float = 1.0,
        target_update_frequency: int = 10_000,
        **kwargs,
    ):
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        # Initialize environment & networks
        self.policy_network = network_factory()
        self.target_network = network_factory()
        # Ensure network's parameters are the same
        self.target_update()
        self.target_network.eval()

        # We can be more efficient with space if we are using discrete environments
        # and don't need to use floats
        if is_discrete_env:
            self.obs_context_type = np.int_
            self.obs_tensor_type = torch.long
        else:
            self.obs_context_type = np.float32
            self.obs_tensor_type = torch.float32

        # PyTorch config
        self.device = device

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env_obs_length=env_obs_length,
            obs_mask=obs_mask,
            max_episode_steps=max_env_steps,
            context_len=context_len,
        )

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_frequency = target_update_frequency

        # Logging
        self.num_train_steps = 0
        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

        self.num_actions = num_actions
        self.train_mode = True
        self.obs_mask = obs_mask
        self.train_context = Context(
            context_len, obs_mask, self.num_actions, env_obs_length
        )
        self.eval_context = Context(
            context_len, obs_mask, self.num_actions, env_obs_length
        )

    @property
    def context(self) -> Context:
        return self.train_context if self.train_mode else self.eval_context

    def eval_on(self) -> None:
        self.train_mode = False
        self.policy_network.eval()

    def eval_off(self) -> None:
        self.train_mode = True
        self.policy_network.train()

    @torch.no_grad()
    def get_action(self, epsilon=0.0) -> int:
        """Use policy_network to get an e-greedy action given the current obs."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.policy_network(
            torch.as_tensor(
                self.context.obs[min(self.context.timestep, self.context_len - 1)],
                dtype=self.obs_tensor_type,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return torch.argmax(q_values).item()

    def observe(self, obs, action, reward, done) -> None:
        if self.train_mode:
            self.replay_buffer.store(obs, action, reward, done)

    def context_reset(self, obs: np.ndarray) -> None:
        self.context.reset(obs)
        if self.train_mode:
            self.replay_buffer.store_obs(obs)

    def train(self) -> None:
        """Perform one gradient step of the network"""
        if not self.replay_buffer.can_sample(self.batch_size):
            return

        self.eval_off()
        obss, actions, rewards, next_obss, dones, _ = self.replay_buffer.sample(
            self.batch_size
        )

        # We pull obss/next_obss as [batch-size x 1 x obs-dim]
        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(
            next_obss, dtype=self.obs_tensor_type, device=self.device
        )
        # Actions is [batch-size x 1 x 1] which we want to be [batch-size x 1]
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        # Rewards/Dones are [batch-size x 1 x 1] which we want to be [batch-size]
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).squeeze()
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device).squeeze()

        # obss is [batch-size x obs-dim] and after network is [batch-size x action-dim]
        # Then we gather it and squeeze to [batch-size]
        q_values = self.policy_network(obss)
        # [batch-seq-actions]
        q_values = q_values.gather(2, actions).squeeze()

        with torch.no_grad():
            # We use DDQN, so the policy network determines which future actions we'd
            # take, but the target network determines the value of those
            next_obs_qs = self.policy_network(next_obss)
            argmax = torch.argmax(next_obs_qs, axis=-1).unsqueeze(-1)
            next_obs_q_values = (
                self.target_network(next_obss).gather(2, argmax).squeeze()
            )

            # here goes BELLMAN
            targets = rewards + (1 - dones) * (next_obs_q_values * self.gamma)

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

    def target_update(self) -> None:
        """Hard update where we copy the network parameters from the policy network to the target network"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_mini_checkpoint(self, checkpoint_dir: str, wandb_id: str) -> None:
        torch.save(
            {"step": self.num_train_steps, "wandb_id": wandb_id},
            checkpoint_dir + "_mini_checkpoint.pt",
        )

    @staticmethod
    def load_mini_checkpoint(checkpoint_dir: str) -> dict:
        return torch.load(checkpoint_dir + "_mini_checkpoint.pt")

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        wandb_id: str,
        episode_rewards: RunningAverage,
        episode_successes: RunningAverage,
        episode_lengths: RunningAverage,
        eps: LinearAnneal,
    ) -> None:
        self.save_mini_checkpoint(checkpoint_dir=checkpoint_dir, wandb_id=wandb_id)
        torch.save(
            # np.savez_compressed(
            # checkpoint_dir + "_checkpoint",
            {
                "step": self.num_train_steps,
                "wandb_id": wandb_id,
                # Replay Buffer
                "replay_buffer_pos": self.replay_buffer.pos,
                # Neural Net
                "policy_net_state_dict": self.policy_network.state_dict(),
                "target_net_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": eps.val,
                # Results
                "episode_rewards": episode_rewards,
                "episode_successes": episode_successes,
                "episode_lengths": episode_lengths,
                # Losses
                "td_errors": self.td_errors,
                "grad_norms": self.grad_norms,
                "qvalue_max": self.qvalue_max,
                "qvalue_mean": self.qvalue_mean,
                "qvalue_min": self.qvalue_min,
                "target_max": self.target_max,
                "target_mean": self.target_mean,
                "target_min": self.target_min,
                # RNG states
                "random_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else torch.get_rng_state(),
            },
            checkpoint_dir + "_checkpoint.pt",
        )
        joblib.dump(self.replay_buffer.obss, checkpoint_dir + "buffer_obss.sav")
        joblib.dump(self.replay_buffer.actions, checkpoint_dir + "buffer_actions.sav")
        joblib.dump(self.replay_buffer.rewards, checkpoint_dir + "buffer_rewards.sav")
        joblib.dump(self.replay_buffer.dones, checkpoint_dir + "buffer_dones.sav")
        joblib.dump(
            self.replay_buffer.episode_lengths, checkpoint_dir + "buffer_eplens.sav"
        )

    def load_checkpoint(
        self, checkpoint_dir: str
    ) -> Tuple[str, RunningAverage, RunningAverage, RunningAverage, float]:
        checkpoint = torch.load(checkpoint_dir + "_checkpoint.pt")
        # checkpoint = np.load(checkpoint_dir + "_checkpoint.npz", allow_pickle=True)

        self.num_train_steps = checkpoint["step"]
        # Replay Buffer
        self.replay_buffer.pos = checkpoint["replay_buffer_pos"]
        self.replay_buffer.obss = joblib.load(checkpoint_dir + "buffer_obss.sav")
        self.replay_buffer.actions = joblib.load(checkpoint_dir + "buffer_actions.sav")
        self.replay_buffer.rewards = joblib.load(checkpoint_dir + "buffer_rewards.sav")
        self.replay_buffer.dones = joblib.load(checkpoint_dir + "buffer_dones.sav")
        self.replay_buffer.episode_lengths = joblib.load(
            checkpoint_dir + "buffer_eplens.sav"
        )
        # Neural Net
        self.policy_network.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Losses
        self.td_errors = checkpoint["td_errors"]
        self.grad_norms = checkpoint["grad_norms"]
        self.qvalue_max = checkpoint["qvalue_max"]
        self.qvalue_mean = checkpoint["qvalue_mean"]
        self.qvalue_min = checkpoint["qvalue_min"]
        self.target_max = checkpoint["target_max"]
        self.target_mean = checkpoint["target_mean"]
        self.target_min = checkpoint["target_min"]
        # RNG states
        random.setstate(checkpoint["random_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["torch_cuda_rng_state"])

        # Results
        episode_rewards = checkpoint["episode_rewards"]
        episode_successes = checkpoint["episode_successes"]
        episode_lengths = checkpoint["episode_lengths"]
        # Exploration value
        epsilon = checkpoint["epsilon"]

        return (
            checkpoint["wandb_id"],
            episode_rewards,
            episode_successes,
            episode_lengths,
            epsilon,
        )
