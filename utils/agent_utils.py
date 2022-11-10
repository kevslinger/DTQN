from dtqn.agents import DtqnAgent, DrqnAgent, DqnAgent, AdrqnAgent
from dtqn.networks import DTQN, DRQN, DQN, ADRQN, DARQN, ATTN
import torch.optim as optim
from utils import env_processing, epsilon_anneal
import gym
import torch
import numpy as np


MODEL_MAP = {
    "DTQN": DTQN,
    "DRQN": DRQN,
    "DQN": DQN,
    "ADRQN": ADRQN,
    "DARQN": DARQN,
    "ATTN": ATTN,
}


def get_agent(
    model_str: str,
    env: gym.Env,
    eval_env: gym.Env,
    embed_per_obs_dim: int,
    inner_embed: int,
    buffer_size: int,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    context_len: int,
    history: bool,
    total_steps: int,
    num_heads: int = 1,
    num_layers: int = 1,
    dropout: float = 0.0,
    identity: bool = True,
    gate: str = "res",
    pos: int = 1,
):
    env_obs_length = env_processing.get_env_obs_length(env)
    env_obs_mask = env_processing.get_env_obs_mask(env)
    if isinstance(env_obs_mask, np.ndarray):
        obs_vocab_size = env_obs_mask.max() + 1
    else:
        obs_vocab_size = env_obs_mask + 1
    is_discrete_env = isinstance(
        env.observation_space,
        (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary),
    )

    def make_model(network_cls):
        return network_cls(
            env_obs_length,
            env.action_space.n,
            embed_per_obs_dim,
            inner_embed,
            is_discrete_env,
            obs_vocab_size=obs_vocab_size,
            batch_size=batch_size,
        ).to(device)

    def make_attn(network_cls):
        return network_cls(
            env_obs_length,
            env.action_space.n,
            embed_per_obs_dim,
            inner_embed,
            num_heads,
            context_len,
            dropout=dropout,
            pos=pos,
            discrete=is_discrete_env,
            vocab_sizes=obs_vocab_size,
        ).to(device)

    def make_dtqn(network_cls):
        return network_cls(
            env_obs_length,
            env.action_space.n,
            embed_per_obs_dim,
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
        ).to(device)

    if "ATTN" in model_str:
        policy_net = make_attn(MODEL_MAP[model_str])
        target_net = make_attn(MODEL_MAP[model_str])
    elif "DTQN" in model_str:
        policy_net = make_dtqn(MODEL_MAP[model_str])
        target_net = make_dtqn(MODEL_MAP[model_str])
    else:
        policy_net = make_model(MODEL_MAP[model_str])
        target_net = make_model(MODEL_MAP[model_str])

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    epsilon_schedule = epsilon_anneal.LinearAnneal(1.0, 0.1, total_steps // 10)

    if MODEL_MAP[model_str] in (ADRQN,):
        agent = AdrqnAgent(
            env,
            eval_env,
            policy_net,
            target_net,
            buffer_size,
            optimizer,
            device,
            env_obs_length,
            epsilon_schedule,
            batch_size=batch_size,
            context_len=context_len,
            embed_size=inner_embed,
            history=history,
        )
    elif MODEL_MAP[model_str] in (DRQN, DARQN):
        agent = DrqnAgent(
            env,
            eval_env,
            policy_net,
            target_net,
            buffer_size,
            optimizer,
            device,
            env_obs_length,
            epsilon_schedule,
            batch_size=batch_size,
            context_len=context_len,
            embed_size=inner_embed,
            history=history,
        )
    elif MODEL_MAP[model_str] in (DQN,):
        agent = DqnAgent(
            env,
            eval_env,
            policy_net,
            target_net,
            buffer_size,
            optimizer,
            device,
            env_obs_length,
            epsilon_schedule,
            batch_size=batch_size,
        )
    elif MODEL_MAP[model_str] in (DTQN, ATTN):
        agent = DtqnAgent(
            env,
            eval_env,
            policy_net,
            target_net,
            buffer_size,
            optimizer,
            device,
            env_obs_length,
            epsilon_schedule,
            batch_size=batch_size,
            context_len=context_len,
            history=history,
        )
    else:
        print("Cannot find that agent/network. Exiting...")
        exit(1)
    return agent
