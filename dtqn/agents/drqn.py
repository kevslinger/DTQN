import torch
from dtqn.agents.dqn import DqnAgent
from utils.env_processing import Context

DEFAULT_CONTEXT_LEN = 50


class DrqnAgent(DqnAgent):
    # noinspection PyTypeChecker
    def __init__(
        self,
        network_factory,
        buf_size: int,
        device: torch.device,
        env_obs_length: int,
        context_len: int = 50,
        embed_size: float = 64,
        history: bool = True,
        obs_mask: float = 0,
        num_actions: int = 3,
        **kwargs
    ):
        super().__init__(
            network_factory,
            buf_size,
            device,
            env_obs_length,
            **kwargs
        )
        self.context_len = context_len
        self.obs_mask = obs_mask
        self.history = history

        zeros_hidden = torch.zeros(
            1,
            1,
            embed_size,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        hidden_states = (zeros_hidden, zeros_hidden)

        self.train_context = Context(context_len, obs_mask, num_actions, hidden_states, env_obs_length)
        self.eval_context = Context(context_len, obs_mask, num_actions, hidden_states, env_obs_length)
        self.context = self.train_context

    def context_reset(self):
        self.context.reset()

    def eval_on(self):
        self.policy_network.eval()
        self.context = self.eval_context

    def eval_off(self):
        self.policy_network.train()
        self.context = self.train_context

    def store_transition(self, cur_obs, obs, action, reward, done, timestep):
        self.context.add(cur_obs, obs, action, reward, done)
        o, o_n, a, r, d = self.context.export()
        if any(action > 3 for action in a): print(a)
        self.replay_buffer.store(o, o_n, a, r, d, min(self.context_len, timestep+1))

    @torch.no_grad()
    def get_action(self, current_obs):
        q_values, self.context.hidden = self.policy_network(
            torch.as_tensor(
                current_obs, dtype=torch.float32, device=self.device
            ).reshape(1, 1, current_obs.shape[0]),
            hidden_states=self.context.hidden,
        )
        return torch.argmax(q_values).item()

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
        obss = torch.as_tensor(obss, dtype=torch.float32, device=self.device)
        next_obss = torch.as_tensor(
            next_obss, dtype=torch.float32, device=self.device
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

        self.qvalue_max.add(q_values.max())
        self.qvalue_mean.add(q_values.mean())
        self.qvalue_min.add(q_values.min())

        self.target_max.add(targets.max())
        self.target_mean.add(targets.mean())
        self.target_min.add(targets.min())

        # Optimization step
        loss = torch.nn.functional.mse_loss(q_values, targets)
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
