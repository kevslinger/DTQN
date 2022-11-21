import torch
from envs.ma_env import MultiAgentEnv
from utils import env_processing, epsilon_anneal


class Team:
    def __init__(
            self,
            env: MultiAgentEnv,
            eval_env: MultiAgentEnv,
            network_factory,
            buffer_size: int,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            exp_coef: epsilon_anneal.LinearAnneal,
            batch_size: int = 32,
            gamma: float = 0.99,
            context_len: int = 1,
            grad_norm_clip: float = 1.0
    ):
        self.env, self.eval_env = env, eval_env
        self.obs_length = env_processing.get_env_obs_length(env)
        self.policy_networks = [network_factory() for _ in range(self.env.n_agents)]
        self.target_networks = [network_factory() for _ in range(self.env.n_agents)]

    def act(self, observation):
        pass

    def feedback(self, observation):
        pass

    def save_mini_checkpoint(self, wandb_id: str, checkpoint_dir: str) -> None:
        pass

    def load_mini_cmeckpoint(self, checkpoint_dir: str) -> dict:
        return torch.load(checkpoint_dir + "_mini_checkpoint.pt")

    def save_checkpoint(self, wandb_id: str, checkpoint_dir: str) -> None:
        pass

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        pass

    def evaluate(self, n_episode=10, render: bool = False) -> None:
        """Evaluate the network for n_episodes"""
        pass

    def target_update(self) -> None:
        pass

    @torch.no_grad()
    def get_action(self, current_obs, epsilon=0.0) -> int:
        pass

    def train(self) -> None:
        """Perform one gradient step of the network"""
        pass

    def step(self) -> bool:
        """Take one step of the environment"""
        pass

    def prepopulate(self, prepop_steps: int) -> None:
        """Prepopulate the replay buffer with `prepop_steps` of experience"""
        pass
