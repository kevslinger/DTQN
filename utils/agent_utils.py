from typing import Tuple
import gym
import torch
import numpy as np

from dtqn.agents.dqn import DqnAgent
from dtqn.agents.drqn import DrqnAgent
from dtqn.agents.dtqn import DtqnAgent
from dtqn.networks.adrqn import ADRQN
from dtqn.networks.drqn import DRQN
from dtqn.networks.darqn import DARQN
from dtqn.networks.dqn import DQN
from dtqn.networks.dtqn import DTQN
from dtqn.networks.perceiver import PerceiverIO
from utils import env_processing


MODEL_MAP = {
    "DTQN": DTQN,
    "Perceiver": PerceiverIO,
    "ADRQN": ADRQN,
    "DRQN": DRQN,
    "DARQN": DARQN,
    "DQN": DQN,
}

AGENT_MAP = {
    "DTQN": DtqnAgent,
    "Perceiver": DtqnAgent,
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
    max_env_steps: int,
    history: int,
    target_update_frequency: int,
    gamma: float,
    num_heads: int = 1,
    num_layers: int = 1,
    dropout: float = 0.0,
    identity: bool = False,
    gate: str = "res",
    pos: str = "learned",
    bag_size: int = 0,
):
    """Function to create the agent. This will also set up the policy and target networks that the agent needs.
    Arguments:
        model_str: str, the name of the Q-function model we are going to use.
        envs: Tuple[gym.Env], a list of gym environments the agent will train and evaluate on. They must all have the same observation and action space.
        ember_per_obs_dim: int, the number of features to give each dimension of the observation. This is only used for discrete domains.
        action_dim: int, the number of features to give each action.
        inner_embed: int, the size of the main transformer model.
        buffer_size: int, the number of transitions to store in the replay buffer.
        device: torch.device, the device to use for training.
        learning_rate: float, the learning rate for the ADAM optimiser.
        batch_size: int, the batch size to use for training.
        context_len: int, the maximum sequence length to use as input to the network.
        max_env_steps: int, the maximum number of steps allowed in the environment before timeout. This will be inferred if not explicitly supplied.
        history: int, the number of Q-values to use during training for each sample.
        target_update_frequency: int, the number of training steps between (hard) target network update.
        gamma: float, the discount factor.
        -DTQN-Specific-
        num_heads: int, the number of heads to use in the MultiHeadAttention.
        num_layers: int, the number of transformer blocks to use.
        dropout: float, the dropout percentage to use.
        identity: bool, whether or not to use identity map reordering.
        gate: str, which combine step to use (residual skip connection or GRU)
        pos: str, which type of position encoding to use ("learned", "sin", or "none")
        bag_size: int, the size of the persistent memory bag

    Returns:
        the agent we created with all those arguments, complete with replay buffer, context, policy and target network.
    """
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
    # Keep the history between 1 and context length
    if history < 1 or history > context_len:
        print(
            f"History must be 1 < history <= context_len, but history is {history} and context len is {context_len}. Clipping history to {np.clip(history, 1, context_len)}..."
        )
        history = np.clip(history, 1, context_len)
    # All envs must share same action space
    num_actions = envs[0].action_space.n

    if model_str == "DQN":
        context_len = 1

    def make_model(network_cls):
        """Creates the non-transformer models: DQN, DRQN, ADRQN, ..."""
        return lambda: network_cls(
            env_obs_length,
            num_actions,
            embed_per_obs_dim,
            action_dim,
            inner_embed,
            is_discrete_env,
            obs_vocab_size=obs_vocab_size,
            batch_size=batch_size,
        ).to(device)

    def make_dtqn(network_cls):
        """Creates DTQN"""
        return lambda: network_cls(
            env_obs_length,
            num_actions,
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
            bag_size=bag_size,
        ).to(device)
    
    def make_perceiver(network_cls):
        """Creates Perceiver"""
        return lambda: network_cls(
            env_obs_length,
            num_actions,
            embed_per_obs_dim,
            action_dim,
            inner_embed,
            num_heads,
            num_layers,
            context_len,
            15,
            inner_embed,
            context_len,
            pos=pos,
            discrete=is_discrete_env,
            vocab_sizes=obs_vocab_size,
            dropout=dropout
        ).to(device)

    if model_str == "Perceiver":
        network_factory = make_perceiver(MODEL_MAP[model_str])
    elif "DTQN" not in model_str:
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
        num_actions,
        is_discrete_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        context_len=context_len,
        embed_size=inner_embed,
        history=history,
        target_update_frequency=target_update_frequency,
        bag_size=bag_size,
    )
