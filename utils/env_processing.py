import gym
from gym import spaces
from gym.wrappers.time_limit import TimeLimit
import numpy as np
from typing import Union

try:
    from gym_gridverse.gym import GymEnvironment
    from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
    from gym_gridverse.outer_env import OuterEnv
    from gym_gridverse.representations.observation_representations import (
        make_observation_representation,
    )
    from gym_gridverse.representations.state_representations import (
        make_state_representation,
    )
except ImportError:
    print(
        f"WARNING: ``gym_gridverse`` is not installed. This means you cannot run an experiment with the `gv_*` domains."
    )
    GymEnvironment = None
from envs.gv_wrapper import GridVerseWrapper
import os
from enum import Enum
from typing import Tuple


class ObsType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


def get_env_obs_type(obs_space: spaces.Space) -> int:
    if isinstance(
        obs_space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)
    ):
        return ObsType.DISCRETE
    else:
        return ObsType.CONTINUOUS


def make_env(id_or_path: str) -> GymEnvironment:
    """Makes a GV gym environment."""
    try:
        print("Loading using gym.make")
        env = gym.make(id_or_path)

    except gym.error.Error:
        print(f"Environment with id {id_or_path} not found.")
        print("Loading using YAML")
        inner_env = factory_env_from_yaml(
            os.path.join(os.getcwd(), "envs", "gridverse", id_or_path)
        )
        state_representation = make_state_representation(
            "default", inner_env.state_space
        )
        observation_representation = make_observation_representation(
            "default", inner_env.observation_space
        )
        outer_env = OuterEnv(
            inner_env,
            state_representation=state_representation,
            observation_representation=observation_representation,
        )
        env = GymEnvironment(outer_env)
        env = TimeLimit(GridVerseWrapper(env), max_episode_steps=500)

    return env


def get_env_obs_length(env: gym.Env) -> int:
    """Gets the length of the observations in an environment"""
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return 1
    elif isinstance(env.observation_space, (gym.spaces.MultiDiscrete, gym.spaces.Box)):
        if len(env.observation_space.shape) != 1:
            raise NotImplementedError(f"We do not yet support 2D observation spaces")
        return env.observation_space.shape[0]
    elif isinstance(env.observation_space, spaces.MultiBinary):
        return env.observation_space.n
    else:
        raise NotImplementedError(f"We do not yet support {env.observation_space}")


def get_env_obs_mask(env: gym.Env) -> Union[int, np.ndarray]:
    """Gets the number of observations possible (for discrete case).
    For continuous case, please edit the -5 to something lower than
    lowest possible observation (while still being finite) so the
    network knows it is padding.
    """
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return env.observation_space.n
    elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        return env.observation_space.nvec + 1
    elif isinstance(env.observation_space, gym.spaces.Box):
        # If you would like to use DTQN with a continuous action space, make sure this value is
        #       below the minimum possible observation. Otherwise it will appear as a real observation
        #       to the network which may cause issues. In our case, Car Flag has min of -1 so this is
        #       fine.
        return -5
    else:
        raise NotImplementedError(f"We do not yet support {env.observation_space}")


def add_to_history(history, obs) -> np.ndarray:
    """Append an observation to the history, and removes the first observation."""
    if isinstance(obs, np.ndarray):
        return np.concatenate((history[1:], [obs.copy()]))
    elif isinstance(obs, (int, float)):
        return np.concatenate((history[1:], [[obs]]))
    # hack to get numpy values
    elif hasattr(obs, "dtype"):
        return np.concatenate((history[1:], [[obs]]))
    else:
        raise ValueError(f"Tried to add to {history} with {obs}, type {obs.dtype}")


def make_empty_contexts(
    context_len: int,
    env_obs_length: int,
    obs_type: type,
    obs_mask: int,
    num_actions: int,
    reward_mask: float,
    done_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_context = np.full([context_len, env_obs_length], obs_mask, dtype=obs_type)
    next_obs_context = np.full([context_len, env_obs_length], obs_mask, dtype=obs_type)
    action_context = np.full(
        [context_len, 1], np.random.randint(num_actions), dtype=np.int_
    )
    reward_context = np.full([context_len, 1], reward_mask, dtype=np.float32)
    done_context = np.full([context_len, 1], done_mask, dtype=np.bool_)
    return obs_context, next_obs_context, action_context, reward_context, done_context
