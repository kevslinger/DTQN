from dtqn.agents.dqn import DqnAgent
import numpy as np
from distutils.log import warn
import torch.nn.functional as F
import torch
import gym

from utils import env_processing, epsilon_anneal
import time


DEFAULT_CONTEXT_LEN = 50


class DrqnAgent(DqnAgent):
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
        context_len: float = 4,
        embed_size: float = 64,
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

        # We use these masks to pad our initial context
        self.obs_mask = env_processing.get_env_obs_mask(self.env)
        if isinstance(self.obs_mask, np.ndarray):
            self.obs_mask = self.obs_mask.max()

        self.reward_mask = 0
        self.done_mask = True

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

        self.timestep = 0
        self.obs_context[self.timestep] = self.current_obs.copy()

        self.zeros_hidden = torch.zeros(
            1,
            1,
            embed_size,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.hidden_states = (self.zeros_hidden.clone(), self.zeros_hidden.clone())

        self.history = history

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

            if self.timestep < self.context_len:
                self.next_obs_context[self.timestep] = obs.copy()
                self.action_context[self.timestep] = action
                self.reward_context[self.timestep] = reward
                self.done_context[self.timestep] = buffer_done
            else:
                self.next_obs_context = env_processing.add_to_history(
                    self.next_obs_context, obs
                )
                self.action_context = env_processing.add_to_history(
                    self.action_context, action
                )
                self.reward_context = env_processing.add_to_history(
                    self.reward_context, reward
                )
                self.done_context = env_processing.add_to_history(
                    self.done_context, buffer_done
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
        self.obs_context[0] = np.array([self.env.reset()]).flatten().copy()

    def step(self) -> bool:
        """Take one step of the environment"""
        action, self.hidden_states = self.get_action(
            self.current_obs, self.hidden_states, epsilon=self.exp_coef.val
        )
        obs, reward, done, info = self.env.step(action)
        obs = np.array([obs]).flatten().copy()

        if info.get("TimeLimit.truncated", False):
            buffer_done = False
        else:
            buffer_done = done

        if self.timestep < self.context_len:
            self.next_obs_context[self.timestep] = obs.copy()
            self.action_context[self.timestep] = action
            self.reward_context[self.timestep] = reward
            self.done_context[self.timestep] = buffer_done
        else:
            self.next_obs_context = env_processing.add_to_history(
                self.next_obs_context, obs
            )
            self.action_context = env_processing.add_to_history(
                self.action_context, action
            )
            self.reward_context = env_processing.add_to_history(
                self.reward_context, reward
            )
            self.done_context = env_processing.add_to_history(
                self.done_context, buffer_done
            )

        self.timestep += 1
        self.replay_buffer.store(
            self.obs_context.copy(),
            self.next_obs_context.copy(),
            self.action_context.copy(),
            self.reward_context.copy(),
            self.done_context.copy(),
            min(self.context_len, self.timestep),  # Episode length
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
            self.current_obs = self.env.reset()
            self.hidden_states = (self.zeros_hidden.clone(), self.zeros_hidden.clone())
        else:
            self.current_obs = obs

        self.current_obs = np.array([self.current_obs]).copy().flatten()

        if self.timestep < self.context_len:
            self.obs_context[self.timestep] = self.current_obs.copy()
        else:
            self.obs_context = env_processing.add_to_history(
                self.obs_context, self.current_obs
            )

        # Anneal epsilon
        self.exp_coef.anneal()

        return done

    @torch.no_grad()
    def get_action(self, current_obs, hidden_states, epsilon=0.0):
        """Use policy_network to get an e-greedy action given the current obs and hidden states (memory)

        If epsilon is not supplied, we take the greedy action (can be used for evaluation)"""
        q_values, hidden = self.policy_network(
            torch.as_tensor(
                current_obs, dtype=self.obs_tensor_type, device=self.device
            ).reshape(1, 1, current_obs.shape[0]),
            hidden_states=hidden_states,
        )
        if np.random.rand() < epsilon:
            # DQN/DRQN action space must be discrete
            return np.random.randint(self.env.action_space.n), hidden
        else:
            return torch.argmax(q_values).item(), hidden

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
                    axis=2,
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
                    axis=1,
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

    def evaluate(self, n_episode: int = 10, render: bool = False) -> None:
        """Evaluate the network for n_episodes"""
        # Set networks to eval mode
        self.policy_network.eval()

        total_reward = 0
        num_successes = 0
        total_steps = 0
        for _ in range(n_episode):
            obs = self.eval_env.reset()
            done = False
            hid_state = (self.zeros_hidden.clone(), self.zeros_hidden.clone())
            timestep = 0
            ep_reward = 0
            if render:
                self.eval_env.render()
                time.sleep(0.5)
            while not done and timestep < self.eval_env_max_steps:
                timestep += 1
                action, hid_state = self.get_action(
                    np.array([obs], dtype=self.obs_context_type).flatten(),
                    hid_state,
                    epsilon=0.0,
                )
                obs, reward, done, info = self.eval_env.step(action)
                ep_reward += reward
                if render:
                    self.eval_env.render()
                    if done:
                        print(f"Episode terminated. Episode reward: {ep_reward}")
                    time.sleep(1)
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
