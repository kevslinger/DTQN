from typing import Tuple
import gym
import torch
import numpy as np

from dtqn.agents.dqn import DqnAgent
from dtqn.agents.adrqn import AdrqnAgent
from dtqn.agents.drqn import DrqnAgent
from dtqn.agents.dtqn import DtqnAgent
from dtqn.networks.adrqn import ADRQN
from dtqn.networks.drqn import DRQN
from dtqn.networks.darqn import DARQN
from dtqn.networks.dqn import DQN
from dtqn.networks.dtqn import DTQN
from dtqn.networks.dtqn_bag import DTQNBag
from utils import env_processing


MODEL_MAP = {
    "DTQN": DTQN,
    "DTQN-bag": DTQNBag,
    "ADRQN": ADRQN,
    "DRQN": DRQN,
    "DARQN": DARQN,
    "DQN": DQN,
}

AGENT_MAP = {
    "DTQN": DtqnAgent,
    "DTQN-bag": DtqnAgent,
    "ADRQN": DrqnAgent,
    "DRQN": DrqnAgent,
    "DARQN": DrqnAgent,
    "DQN": DqnAgent,
}


def get_agent(
    model_str: str,
    envs: Tuple[gym.Env],
    embed_per_obs_dim: int,
    action_dim: int,
    inner_embed: int,
    buffer_size: int,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    context_len: int,
    eval_context_len: int,
    max_env_steps: int,
    history: bool,
    target_update_frequency: int,
    gamma: float,
    num_heads: int = 1,
    num_layers: int = 1,
    dropout: float = 0.0,
    identity: bool = False,
    gate: str = "res",
    pos: int = 1,
    bag_size: int = 0,
    eval_bag_size: int = 0,
):
    # All envs must have the same observation shape
    env_obs_length = env_processing.get_env_obs_length(envs[0])
    env_obs_mask = env_processing.get_env_obs_mask(envs[0])
    if max_env_steps <= 0:
        max_env_steps = max([env_processing.get_env_max_steps(env) for env in envs])
    if isinstance(env_obs_mask, np.ndarray):
        obs_vocab_size = env_obs_mask.max() + 1
    else:
        obs_vocab_size = env_obs_mask + 1
    is_discrete_env = isinstance(
        envs[0].observation_space,
        (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary),
    )

    if model_str == "DQN":
        context_len = 1

    def make_model(network_cls):
        return lambda: network_cls(
            env_obs_length,
            envs[0].action_space.n,
            embed_per_obs_dim,
            action_dim,
            inner_embed,
            is_discrete_env,
            obs_vocab_size=obs_vocab_size,
            batch_size=batch_size,
        ).to(device)

    def make_dtqn(network_cls):
        return lambda: network_cls(
            env_obs_length,
            envs[0].action_space.n,
            embed_per_obs_dim,
            action_dim,
            inner_embed,
            num_heads,
            num_layers,
            context_len,
            dropout=dropout,
            gate=gate,
            identity=identity,
            pos=pos,
            discrete=is_discrete_env,
            vocab_sizes=obs_vocab_size,
            target_update_frequency=target_update_frequency,
            bag_size=bag_size,
        ).to(device)

    if "DTQN" not in model_str:
        network_factory = make_model(MODEL_MAP[model_str])
    else:
        network_factory = make_dtqn(MODEL_MAP[model_str])

    return AGENT_MAP[model_str](
        network_factory,
        buffer_size,
        device,
        env_obs_length,
        max_env_steps,
        env_obs_mask,
        envs[0].action_space.n,
        is_discrete_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        context_len=context_len,
        eval_context_len=eval_context_len,
        embed_size=inner_embed,
        history=history,
        target_update_frequency=target_update_frequency,
        bag_size=bag_size,
        eval_bag_size=eval_bag_size,
    )
