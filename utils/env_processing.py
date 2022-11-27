from collections import deque

import torch
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


# noinspection PyAttributeOutsideInit
class Context:
    def __init__(self, length: int, obs_mask, num_actions, initial_hidden, env_obs_length):
        self.length = length
        self.env_obs_length = env_obs_length
        self.num_actions = num_actions
        self.obs_mask = obs_mask
        self.reward_mask = 0
        self.done_mask = True
        self.initial_hidden = initial_hidden
        self.reset()

    def reset(self):
        self.obs = deque([[self.obs_mask]*self.env_obs_length]*self.length, maxlen=self.length)
        self.next_obs = deque([[self.obs_mask]*self.env_obs_length]*self.length, maxlen=self.length)
        self.action = deque([[np.random.randint(self.num_actions)]]*self.length, maxlen=self.length)
        self.reward = deque([[self.reward_mask]]*self.length, maxlen=self.length)
        self.done = deque([[self.done_mask]]*self.length, maxlen=self.length)
        self.hidden = self.initial_hidden

    def add(self, o, next_o, a, r, done):
        self.obs.append(o)
        self.next_obs.append(next_o)
        self.action.append([a])
        self.reward.append([r])
        self.done.append([done])

    def export(self):
        return (
            np.array(self.obs),
            np.array(self.next_obs),
            np.array(self.action),
            np.array(self.reward),
            np.array(self.done),
        )

    def get_history_of(self, obs):
        res = self.obs.copy()
        res.append(obs)
        return np.array(res)

