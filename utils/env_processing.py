from collections import deque

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
        return max(env.observation_space.nvec) + 1
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
    """A Dataclass dedicated to storing the agent's history (up to the previous `max_length` transitions)

    Args:
        context_length: The maximum number of transitions to store
        obs_mask: The mask to use for observations not yet seen
        num_actions: The number of possible actions we can take in the environment
        env_obs_length: The dimension of the observations (assume 1d arrays)
    """

    def __init__(self, context_length: int, obs_mask, num_actions, env_obs_length, init_hidden=None):
        self.max_length = context_length
        self.env_obs_length = env_obs_length
        self.num_actions = num_actions
        self.obs_mask = obs_mask
        self.reward_mask = 0
        self.done_mask = True
        self.timestep = 0
        self.init_hidden = init_hidden
        self.hidden = init_hidden
        self.reset()

    def reset(self):
        """Resets to a fresh context"""
        self.obs = np.array(
            [np.array([self.obs_mask] * self.env_obs_length)] * self.max_length,
        )
        self.next_obs = np.array(
            [np.array([self.obs_mask] * self.env_obs_length)] * self.max_length,
        )
        self.action = np.array(
            [np.array([np.random.randint(self.num_actions)])] * self.max_length,
        )
        self.reward = np.array(
            [np.array([self.reward_mask])] * self.max_length,
        )
        self.done = np.array(
            [np.array([self.done_mask])] * self.max_length,
        )
        self.hidden = None
        self.timestep = 0

    def add_transition(self, o: np.ndarray, next_o: np.ndarray, a: int, r: float, done: bool):
        """Complete the transition with the next observation, action, reward, and done flag. If the context is full,
        evict the oldest information """
        t = self.timestep if self.timestep < self.max_length else 0
        self.obs[t] = o
        self.next_obs[t] = next_o
        self.action[t] = np.array([a])
        self.reward[t] = np.array([r])
        self.done[t] = np.array([done])
        if self.timestep >= self.max_length:
            self.obs = self.roll(self.obs)
            self.next_obs = self.roll(self.next_obs)
            self.action = self.roll(self.action)
            self.reward = self.roll(self.reward)
            self.done = self.roll(self.done)
        self.timestep += 1

    def export(
            self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Export the context"""
        return (
            self.obs,
            self.next_obs,
            self.action,
            self.reward,
            self.done,
        )

    @staticmethod
    def roll(arr: np.ndarray):
        return np.roll(arr, -1, axis=0)

    @property
    def last_action(self):
        """Get the last action taken"""
        return self.action[-1][0]

    @property
    def obs_history(self):
        """Get the agent's observation history.

        NOTE: We typically use this once we've seen an observation but before completing a context. That's why we're
        using `self.timestep+1` """
        return self.obs[: self.timestep + 1]

    @staticmethod
    def context_like(context):
        """Creates a new context to mimic the supplied context"""
        return Context(
            context.max_length,
            context.obs_mask,
            context.num_actions,
            context.env_obs_length,
            init_hidden=context.init_hidden,
        )
