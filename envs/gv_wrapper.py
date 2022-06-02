import gym
import gym.spaces
from typing import Union, Optional
import numpy as np
from gym.utils import seeding


class GridVerseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.MultiDiscrete(
            [
                env.observation_space["grid"].high.max() * 2 + 1
                for _ in range(
                    env.observation_space["grid"].shape[0]
                    * env.observation_space["grid"].shape[1]
                )
            ]
        )

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        obs = self.env.reset()
        return (obs["grid"][:, :, 0] + obs["grid"][:, :, 2]).flatten()

    def step(self, action: Union[int, np.ndarray]):
        obs, reward, done, info = self.env.step(action)
        obs = (obs["grid"][:, :, 0] + obs["grid"][:, :, 2]).flatten()
        return obs, reward, done, info
