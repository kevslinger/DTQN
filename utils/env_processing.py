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

from utils.random import RNG


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
        env = TimeLimit(GridVerseWrapper(env), max_episode_steps=250)

    return env


class ObsType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    IMAGE = 2


def get_env_obs_type(env: gym.Env) -> int:
    obs_space = env.observation_space
    sample_obs = env.reset()
    # Check for image first
    if (
        (isinstance(sample_obs, np.ndarray) and len(sample_obs.shape) == 3)
        and isinstance(obs_space, spaces.Box)
        and np.all(obs_space.low == 0)
        and np.all(obs_space.high == 255)
    ):
        return ObsType.IMAGE
    elif isinstance(
        obs_space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)
    ):
        return ObsType.DISCRETE
    else:
        return ObsType.CONTINUOUS


def get_env_obs_length(env: gym.Env) -> int:
    """Gets the length of the observations in an environment"""
    if get_env_obs_type(env) == ObsType.IMAGE:
        return env.reset().shape
    elif isinstance(env.observation_space, gym.spaces.Discrete):
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
    # Check image first
    if get_env_obs_type(env) == ObsType.IMAGE:
        return 0
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return env.observation_space.n
    elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        return max(env.observation_space.nvec) + 1
    elif isinstance(env.observation_space, gym.spaces.Box):
        # If you would like to use DTQN with a continuous action space, make sure this value is
        #       below the minimum possible observation. Otherwise it will appear as a real observation
        #       to the network which may cause issues. In our case, Car Flag has min of -1 so this is
        #       fine.
        return -5
    else:
        raise NotImplementedError(f"We do not yet support {env.observation_space}")


def get_env_max_steps(env: gym.Env) -> Union[int, None]:
    """Gets the maximum steps allowed in an episode before auto-terminating"""
    try:
        return env._max_episode_steps
    except AttributeError:
        try:
            return env.max_episode_steps
        except AttributeError:
            return None
