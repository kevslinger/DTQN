import numpy as np
import torch.nn.functional as F
import torch
from dtqn.agents.drqn import DrqnAgent
from utils.env_processing import Context


class DtqnAgent(DrqnAgent):
    @torch.no_grad()
    def get_action(self, obs, epsilon: float = 0.0) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        # the policy network gets [1, timestep+1 x obs length] as input and
        # outputs [1, timestep+1 x 4 outputs]
        q_values = self.policy_network(
            torch.as_tensor(
                self.context.hist_with_obs(obs)[:self.context.timestep+1],
                dtype=self.obs_tensor_type,
                device=self.device,
            ).unsqueeze(0)
        )
        # We take the argmax of the last timestep's Q values
        # In other words, select the highest q value action
        return torch.argmax(q_values[:, -1, :]).item()

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
