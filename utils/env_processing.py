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


# noinspection PyAttributeOutsideInit
class Context:
    """A Dataclass dedicated to storing the agent's history (up to the previous `max_length` transitions)

    Args:
        context_length: The maximum number of transitions to store
        obs_mask: The mask to use for observations not yet seen
        num_actions: The number of possible actions we can take in the environment
        env_obs_length: The dimension of the observations (assume 1d arrays)
        init_hidden: The initial value of the hidden states (used for RNNs)
    """

    def __init__(
        self,
        context_length: int,
        obs_mask: int,
        num_actions: int,
        env_obs_length: int,
        init_hidden=None,
    ):
        self.max_length = context_length
        self.env_obs_length = env_obs_length
        self.num_actions = num_actions
        self.obs_mask = obs_mask
        self.reward_mask = 0.0
        self.done_mask = True
        self.timestep = 0
        self.init_hidden = init_hidden

    def reset(self, obs: np.ndarray):
        """Resets to a fresh context"""
        if isinstance(self.env_obs_length, tuple):
            self.obs = np.full(
                [self.max_length + 1, *self.env_obs_length],
                self.obs_mask,
                dtype=np.uint8,
            )
        else:
            self.obs = np.full(
                [self.max_length + 1, self.env_obs_length], self.obs_mask
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
        self.hidden = self.init_hidden
        self.timestep = 0
        # Initial observation
        self.obs[0] = obs

    def add_transition(self, o: np.ndarray, a: int, r: float, done: bool) -> None:
        """Add an entire transition. If the context is full, evict the oldest transition"""
        # TODO: need to roll specially because it is 1 longer
        self.obs = (
            np.roll(self.obs, -1, axis=0)
            if self.timestep >= self.max_length + 1
            else self.obs
        )
        self.action = self.roll(self.action)
        self.reward = self.roll(self.reward)
        self.done = self.roll(self.done)

        t = min(self.timestep, self.max_length - 1)
        self.obs[t + 1] = o
        self.action[t] = np.array([a])
        self.reward[t] = np.array([r])
        self.done[t] = np.array([done])
        self.timestep += 1

    def export(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Export the context"""
        current_timestep = min(self.timestep, self.max_length) - 1
        return (
            self.obs[current_timestep],
            # self.next_obs[current_timestep],
            self.action[current_timestep],
            self.reward[current_timestep],
            self.done[current_timestep],
        )

    def hist_with_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return an observation appended to the observation history, but do not commit it"""
        hist = self.roll(self.obs).copy()
        hist[min(self.timestep, self.max_length - 1)] = obs
        return hist

    def roll(self, arr: np.ndarray) -> np.ndarray:
        """Utility function to help with insertions at the end of the array. If the context is full, we replace the first element with the new element, then 'roll' the new element to the end of the array"""
        return np.roll(arr, -1, axis=0) if self.timestep >= self.max_length else arr

    def update_hidden(self, hidden):
        """Replace the hidden state (for use with RNNs)"""
        self.hidden = hidden

    @property
    def last_action(self):
        """Get the last action taken"""
        return self.action[-1][0]

    @property
    def obs_history(self):
        """Get the agent's observation history.

        NOTE: We typically use this once we've seen an observation but before completing a context. That's why we're using `self.timestep+1`"""
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
