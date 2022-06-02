from distutils.log import warn
from typing import Optional
from dtqn.agents.dqn import DqnAgent
from utils import env_processing, epsilon_anneal
import numpy as np
import torch.nn.functional as F
import torch
import gym
from gym import spaces
import time
import random


DEFAULT_CONTEXT_LEN = 50


class DtqnAgent(DqnAgent):
    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy_network,
        target_network,
        buf_size: int,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        env_obs_length: int,
        exp_coef: epsilon_anneal.LinearAnneal,
        batch_size: int = 32,
        gamma: float = 0.99,
        context_len: Optional[int] = None,
        grad_norm_clip: float = 1.0,
        history: bool = True,
    ):

        # Get max sequence length
        if context_len is not None:
            context_len = context_len
        else:
            # If not supplied explicitly, we need to infer
            try:
                context_len = self.env._max_episode_steps
                if context_len is None:
                    context_len = DEFAULT_CONTEXT_LEN
            except AttributeError:
                context_len = DEFAULT_CONTEXT_LEN
                warn(
                    f"Environment {env} does not have a max episode steps. Either "
                    f"explicitly provide a `context_len` or wrap the env in a "
                    f"`gym.wrappers.TimeLimit` wrapper"
                )

        super().__init__(
            env,
            eval_env,
            policy_network,
            target_network,
            buf_size,
            optimizer,
            device,
            env_obs_length,
            exp_coef,
            batch_size=batch_size,
            gamma=gamma,
            context_len=context_len,
            grad_norm_clip=grad_norm_clip,
        )

        # Get observation shape and mask
        self.env_obs_length = env_processing.get_env_obs_length(self.env)
        self.obs_mask = env_processing.get_env_obs_mask(self.env)
        self.reward_mask = 0
        self.done_mask = True

        # Hyperparams
        self.history = history

        # We send the entire history through the network to get Q values
        # The history will be initialized like [inf, inf, inf, ...]
        # As the agent gets observations, it will become [obs1, obs2, inf, inf, ...]
        # And next_obs array will be [obs2, obs3, inf, ...]
        (
            self.obs_context,
            self.next_obs_context,
            self.action_context,
            self.reward_context,
            self.done_context,
        ) = env_processing.make_empty_contexts(
            self.context_len,
            self.env_obs_length,
            self.obs_context_type,
            self.obs_mask,
            self.env.action_space.n,
            self.reward_mask,
            self.done_mask,
        )

        # Reset the environment
        self.timestep = 0

        self.obs_context[self.timestep] = np.array([self.env.reset()]).flatten().copy()

    def prepopulate(self, prepop_steps: int) -> None:
        """Prepopulate the replay buffer with `prepop_steps` of experience"""
        episode_timestep = 0
        for _ in range(prepop_steps):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            obs = np.array([obs]).flatten()
            episode_timestep += 1

            if info.get("TimeLimit.truncated", False):
                buffer_done = False
            else:
                buffer_done = done

            if episode_timestep < self.context_len:
                self.action_context[episode_timestep] = action
                self.reward_context[episode_timestep] = reward
                self.done_context[episode_timestep] = buffer_done
                self.next_obs_context[episode_timestep] = obs.copy()
            else:
                self.action_context = env_processing.add_to_history(
                    self.action_context, action
                )
                self.reward_context = env_processing.add_to_history(
                    self.reward_context, reward
                )
                self.done_context = env_processing.add_to_history(
                    self.done_context, buffer_done
                )
                self.next_obs_context = env_processing.add_to_history(
                    self.next_obs_context, obs
                )
            self.replay_buffer.store(
                self.obs_context.copy(),
                self.next_obs_context.copy(),
                self.action_context.copy(),
                self.reward_context.copy(),
                self.done_context.copy(),
                episode_length=min(self.context_len, episode_timestep),
            )

            if done:
                episode_timestep = 0
                (
                    self.obs_context,
                    self.next_obs_context,
                    self.action_context,
                    self.reward_context,
                    self.done_context,
                ) = env_processing.make_empty_contexts(
                    self.context_len,
                    self.env_obs_length,
                    self.obs_context_type,
                    self.obs_mask,
                    self.env.action_space.n,
                    self.reward_mask,
                    self.done_mask,
                )
                obs = np.array([self.env.reset()]).flatten()

            if episode_timestep < self.context_len:
                self.obs_context[episode_timestep] = obs.copy()
            else:
                self.obs_context = env_processing.add_to_history(self.obs_context, obs)
        # Reset all contexts after loop
        (
            self.obs_context,
            self.next_obs_context,
            self.action_context,
            self.reward_context,
            self.done_context,
        ) = env_processing.make_empty_contexts(
            self.context_len,
            self.env_obs_length,
            self.obs_context_type,
            self.obs_mask,
            self.env.action_space.n,
            self.reward_mask,
            self.done_mask,
        )
        obs = np.array([self.env.reset()]).flatten()
        self.obs_context[0] = obs.copy()

    def step(self) -> bool:
        """Take one step of the environment"""
        context = self.obs_context[: self.timestep + 1]
        action = self.get_action(context, epsilon=self.exp_coef.val)
        obs, reward, done, info = self.env.step(action)
        obs = np.array([obs]).flatten()

        # If the episode gets truncated due to a max time limit, we don't want to store
        # That transition as being "done" (it technically is not)
        if info.get("TimeLimit.truncated", False):
            buffer_done = False
        else:
            buffer_done = done

        # Add transition to buffers. Note that we do not update our obs_context buffer until after storing
        # in replay memory
        if self.timestep < self.context_len:
            self.action_context[self.timestep] = action
            self.reward_context[self.timestep] = reward
            self.done_context[self.timestep] = buffer_done
            self.next_obs_context[self.timestep] = obs.copy()
        else:
            self.action_context = env_processing.add_to_history(
                self.action_context, action
            )
            self.reward_context = env_processing.add_to_history(
                self.reward_context, reward
            )
            self.done_context = env_processing.add_to_history(
                self.done_context, buffer_done
            )
            self.next_obs_context = env_processing.add_to_history(
                self.next_obs_context, obs
            )

        self.timestep += 1
        self.replay_buffer.store(
            self.obs_context.copy(),
            self.next_obs_context.copy(),
            self.action_context.copy(),
            self.reward_context.copy(),
            self.done_context.copy(),
            episode_length=min(self.context_len, self.timestep),
        )

        if done:
            self.current_episode += 1
            self.timestep = 0
            (
                self.obs_context,
                self.next_obs_context,
                self.action_context,
                self.reward_context,
                self.done_context,
            ) = env_processing.make_empty_contexts(
                self.context_len,
                self.env_obs_length,
                self.obs_context_type,
                self.obs_mask,
                self.env.action_space.n,
                self.reward_mask,
                self.done_mask,
            )
            obs = np.array([self.env.reset()]).flatten()

        if self.timestep < self.context_len:
            self.obs_context[self.timestep] = obs.copy()
        else:
            self.obs_context = env_processing.add_to_history(self.obs_context, obs)

        # Anneal epsilon
        self.exp_coef.anneal()

        return done

    @torch.no_grad()
    def get_action(self, history, epsilon=0.0) -> int:
        """Use policy network to get an e-greedy action given the current obs"""
        if np.random.rand() < epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            # the policy network gets [1, timestep+1 x obs length] as input and
            # outputs [1, timestep+1 x 4 outputs]
            q_values = self.policy_network(
                torch.as_tensor(
                    history, dtype=self.obs_tensor_type, device=self.device
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
        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(
            next_obss, dtype=self.obs_tensor_type, device=self.device
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
                argmax = torch.argmax(self.policy_network(next_obss), axis=2).unsqueeze(
                    -1
                )
                next_obs_q_values = self.target_network(next_obss)
                next_obs_q_values = next_obs_q_values.gather(2, argmax).squeeze()
            else:
                argmax = torch.argmax(
                    self.policy_network(next_obss)[:, -1, :], axis=1
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

        self.qvalue_max[self.num_steps % len(self.qvalue_max)] = q_values.max()
        self.qvalue_mean[self.num_steps % len(self.qvalue_mean)] = q_values.mean()
        self.qvalue_min[self.num_steps % len(self.qvalue_min)] = q_values.min()
        self.target_max[self.num_steps % len(self.target_max)] = targets.max()
        self.target_mean[self.num_steps % len(self.target_mean)] = targets.mean()
        self.target_min[self.num_steps % len(self.target_min)] = targets.min()

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

    def evaluate(self, n_episode: int = 10, render: bool = False) -> None:
        """Evaluate the network for n_episodes"""
        # Set networks to eval mode (turns off dropout, etc.)
        self.policy_network.eval()

        total_reward = 0
        num_successes = 0
        total_steps = 0

        for _ in range(n_episode):
            history = np.full([self.context_len, self.env_obs_length], self.obs_mask)
            obs = self.eval_env.reset()
            done = False
            timestep = 0
            ep_reward = 0
            if render:
                self.eval_env.render()
                time.sleep(0.5)
            while not done and timestep < self.eval_env_max_steps:
                timestep += 1
                history = env_processing.add_to_history(history, obs)
                action = self.get_action(
                    history[-min(self.context_len, timestep) :], epsilon=0.0
                )

                obs, reward, done, info = self.eval_env.step(action)
                ep_reward += reward
                if render:
                    self.eval_env.render()
                    if done:
                        print(f"Episode terminated. Episode reward: {ep_reward}")
                    time.sleep(0.5)
            total_reward += ep_reward

            if info.get("is_success", False) or ep_reward > 0:
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
