import numpy as np
import torch
import torch.nn.functional as F
from dtqn.buffers.replay_buffer import ReplayBuffer
from utils import epsilon_anneal
import gym
from gym import spaces
import time
import random


DEFAULT_CONTEXT_LEN = 50


class DqnAgent:
    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy_network,
        target_network,
        buffer_size: int,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        env_obs_length: int,
        exp_coef: epsilon_anneal.LinearAnneal,
        batch_size: int = 32,
        gamma: float = 0.99,
        context_len: int = 1,
        grad_norm_clip: float = 1.0,
    ):
        # Initialize environment & networks
        self.env = env
        self.env_obs_length = env_obs_length
        self.eval_env = eval_env
        self.context_len = context_len
        try:
            self.eval_env_max_steps = self.eval_env._max_episode_steps
        except AttributeError:
            self.eval_env_max_steps = self.context_len * 100
        is_discrete_env = isinstance(
            self.env.observation_space,
            (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        )
        if is_discrete_env:
            self.obs_context_type = np.int_
            self.obs_tensor_type = torch.long
        else:
            self.obs_context_type = np.float32
            self.obs_tensor_type = torch.float32

        self.policy_network = policy_network
        self.target_network = target_network
        self.target_network.eval()
        # Ensure network's parameters are the same
        self.target_update()

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env_obs_length=self.env_obs_length,
            context_len=context_len,
        )
        self.optimizer = optimizer
        self.device = device

        # Hyperparameters
        self.batch_size = batch_size
        self.exp_coef = exp_coef
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip

        # Logging
        self.current_episode = 0
        self.num_steps = 0
        self.num_evals = 0
        self.episode_rewards = np.zeros(10, dtype=np.float32)
        self.episode_successes = np.zeros(10, dtype=np.float32)
        self.episode_lengths = np.zeros(10, dtype=np.float32)
        self.td_errors = np.zeros(100, dtype=np.float32)
        self.grad_norms = np.zeros(100, dtype=np.float32)
        self.qvalue_max = np.zeros(100, dtype=np.float32)
        self.target_max = np.zeros(100, dtype=np.float32)
        self.qvalue_mean = np.zeros(100, dtype=np.float32)
        self.target_mean = np.zeros(100, dtype=np.float32)
        self.qvalue_min = np.zeros(100, dtype=np.float32)
        self.target_min = np.zeros(100, dtype=np.float32)

        # Reset the environment
        self.current_obs = np.array([self.env.reset()]).flatten().copy()

    def prepopulate(self, prepop_steps: int) -> None:
        """Prepopulate the replay buffer with `prepop_steps` of experience"""
        for _ in range(prepop_steps):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            obs = np.array([obs]).flatten().copy()

            if info.get("TimeLimit.truncated", False):
                buffer_done = False
            else:
                buffer_done = done

            self.replay_buffer.store(
                self.current_obs.copy(),
                obs.copy(),
                np.array([action]),
                np.array([reward]),
                np.array([buffer_done]),
            )

            if done:
                obs = self.env.reset()
            self.current_obs = np.array([obs]).flatten().copy()
        obs = np.array([self.env.reset()]).flatten()
        self.current_obs = obs.copy()

    def step(self) -> bool:
        """Take one step of the environment"""
        action = self.get_action(
            np.expand_dims(self.current_obs, axis=0), epsilon=self.exp_coef.val
        )
        obs, reward, done, info = self.env.step(action)
        obs = np.array([obs]).flatten().copy()

        if info.get("TimeLimit.truncated", False):
            buffer_done = False
        else:
            buffer_done = done

        self.replay_buffer.store(
            self.current_obs.copy(),
            obs.copy(),
            np.array([action]),
            np.array([reward]),
            np.array([buffer_done]),
        )

        if done:
            self.current_episode += 1
            obs = self.env.reset()

        self.current_obs = np.array([obs]).flatten().copy()

        # Anneal epsilon
        self.exp_coef.anneal()

        return done

    @torch.no_grad()
    def get_action(self, current_obs, epsilon=0.0) -> int:
        """Use policy_network to get an e-greedy action given the current obs.

        If epsilon is not supplied, we take the greedy action (can be used for evaluation)
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.env.action_space.n)
        q_values = self.policy_network(
            torch.as_tensor(current_obs, dtype=self.obs_tensor_type, device=self.device)
        )
        return torch.argmax(q_values).item()

    def train(self) -> None:
        """Perform one gradient step of the network"""
        if not self.replay_buffer.can_sample(self.batch_size):
            return
        obss, actions, rewards, next_obss, dones, _ = self.replay_buffer.sample(
            self.batch_size
        )

        # We pull obss/next_obss as [batch-size x 1 x obs-dim] so we need to squeeze it
        obss = torch.as_tensor(
            obss, dtype=self.obs_tensor_type, device=self.device
        ).squeeze()
        next_obss = torch.as_tensor(
            next_obss, dtype=self.obs_tensor_type, device=self.device
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

        self.qvalue_max[self.num_steps % len(self.qvalue_max)] = q_values.max()
        self.qvalue_mean[self.num_steps % len(self.qvalue_mean)] = q_values.mean()
        self.qvalue_min[self.num_steps % len(self.qvalue_min)] = q_values.min()

        self.target_max[self.num_steps % len(self.target_max)] = targets.max()
        self.target_mean[self.num_steps % len(self.target_mean)] = targets.mean()
        self.target_min[self.num_steps % len(self.target_min)] = targets.min()

        # Optimization step
        loss = F.mse_loss(q_values, targets)
        self.td_errors[self.num_steps % len(self.td_errors)] = loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.grad_norm_clip,
            error_if_nonfinite=True,
        )
        self.grad_norms[self.num_steps % len(self.grad_norms)] = norm
        self.optimizer.step()
        self.num_steps += 1

    def target_update(self) -> None:
        """Hard update where we copy the network parameters from the policy network to the target network"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def evaluate(self, n_episode=10, render: bool = False) -> None:
        """Evaluate the network for n_episodes"""
        # Set networks to eval mode
        self.policy_network.eval()

        total_reward = 0
        num_successes = 0
        total_steps = 0
        for _ in range(n_episode):
            obs = self.eval_env.reset()
            done = False
            timestep = 0
            ep_reward = 0
            if render:
                self.eval_env.render()
                time.sleep(0.5)
            while not done and timestep < self.eval_env_max_steps:
                timestep += 1
                action = self.get_action(np.expand_dims(obs, axis=0), epsilon=0.0)
                obs, reward, done, info = self.eval_env.step(action)
                ep_reward += reward
                if render:
                    self.eval_env.render()
                    if done:
                        print(f"Episode terminated. Episode reward: {ep_reward}")
                    time.sleep(1)
            total_reward += ep_reward
            if info.get("is_success") or ep_reward > 0:
                num_successes += 1
            total_steps += timestep
        self.eval_env.reset()
        self.episode_successes[self.num_evals % len(self.episode_successes)] = (
            num_successes / n_episode
        )
        self.episode_rewards[self.num_evals % len(self.episode_rewards)] = (
            total_reward / n_episode
        )
        self.episode_lengths[self.num_evals % len(self.episode_lengths)] = (
            total_steps / n_episode
        )
        self.num_evals += 1
        # Set networks back to train mode
        self.policy_network.train()
        return (
            num_successes / n_episode,
            total_reward / n_episode,
            total_steps / n_episode,
        )

    def save_mini_checkpoint(self, wandb_id: str, checkpoint_dir: str) -> None:
        torch.save(
            {"step": self.num_steps, "wandb_id": wandb_id},
            checkpoint_dir + "_mini_checkpoint.pt",
        )

    def load_mini_checkpoint(self, checkpoint_dir: str) -> dict:
        return torch.load(checkpoint_dir + "_mini_checkpoint.pt")

    def save_checkpoint(self, wandb_id: str, checkpoint_dir: str) -> None:
        self.save_mini_checkpoint(wandb_id=wandb_id, checkpoint_dir=checkpoint_dir)
        torch.save(
            {
                "step": self.num_steps,
                "episode": self.current_episode,
                "eval": self.num_evals,
                "wandb_id": wandb_id,
                # Replay Buffer
                "replay_buffer_pos": self.replay_buffer.pos,
                "replay_buffer_obss": self.replay_buffer.obss,
                "replay_buffer_next_obss": self.replay_buffer.next_obss,
                "replay_buffer_actions": self.replay_buffer.actions,
                "replay_buffer_rewards": self.replay_buffer.rewards,
                "replay_buffer_dones": self.replay_buffer.dones,
                "replay_buffer_episode_lengths": self.replay_buffer.episode_lengths,
                # Neural Net
                "policy_net_state_dict": self.policy_network.state_dict(),
                "target_net_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.exp_coef.val,
                # Results
                "episode_rewards": self.episode_rewards,
                "episode_successes": self.episode_successes,
                "episode_lengths": self.episode_lengths,
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

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint = torch.load(checkpoint_dir + "_checkpoint.pt")

        self.num_steps = checkpoint["step"]
        self.current_episode = checkpoint["episode"]
        self.num_evals = checkpoint["eval"]
        # Replay Buffer
        self.replay_buffer.pos = checkpoint["replay_buffer_pos"]
        self.replay_buffer.obss = checkpoint["replay_buffer_obss"]
        self.replay_buffer.next_obss = checkpoint["replay_buffer_next_obss"]
        self.replay_buffer.actions = checkpoint["replay_buffer_actions"]
        self.replay_buffer.rewards = checkpoint["replay_buffer_rewards"]
        self.replay_buffer.dones = checkpoint["replay_buffer_dones"]
        self.replay_buffer.episode_lengths = checkpoint["replay_buffer_episode_lengths"]
        # Neural Net
        self.policy_network.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Results
        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_successes = checkpoint["episode_successes"]
        self.episode_lengths = checkpoint["episode_lengths"]
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

        return checkpoint["wandb_id"]
