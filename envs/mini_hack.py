import gym
import gym.spaces
from typing import Union, Optional
import numpy as np
from gym.utils import seeding


GLYPHS = "glyphs_crop"
PIXEL = "pixel_crop"


def reshape(obs: np.ndarray) -> np.ndarray:
    """Reshape image from HxWxC to CxHxW"""
    obs_shape = obs.shape
    return obs.reshape(obs_shape[2], obs_shape[0], obs_shape[1])


class MiniHackWrapper(gym.Wrapper):
    def __init__(
        self,
        env_id: str,
        obs_type: str = GLYPHS,
        obs_crop: int = 9,
        des_file: str = None,
    ):
        if des_file:
            env = gym.make(
                "MiniHack-Navigation-Custom-v0",
                des_file=des_file,
                observation_keys=(obs_type,),
                obs_crop_h=obs_crop,
                obs_crop_w=obs_crop,
            )
        else:
            env = gym.make(
                env_id,
                observation_keys=(obs_type,),
                obs_crop_h=obs_crop,
                obs_crop_w=obs_crop,
            )
        super().__init__(env)
        self.obs_type = obs_type
        # env.observation_space['glyphs_crop'].shape = (7, 7)
        if self.obs_type == GLYPHS:
            self.observation_space = gym.spaces.MultiDiscrete(
                [
                    env.observation_space[obs_type].high.max()
                    for _ in range(
                        env.observation_space[obs_type].shape[0]
                        * env.observation_space[obs_type].shape[1]
                    )
                ]
            )
        elif self.obs_type == PIXEL:
            obs_shape = env.observation_space[obs_type].shape
            self.observation_space = gym.spaces.Box(
                0, 255, (obs_shape[2], obs_shape[0], obs_shape[1]), np.uint8
            )

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        obs = self.env.reset()
        if self.obs_type == GLYPHS:
            return obs[self.obs_type].flatten()
        elif self.obs_type == PIXEL:
            obs = reshape(obs[self.obs_type])
            return obs

    def step(self, action: Union[int, np.ndarray]):
        obs, reward, done, info = self.env.step(action)
        if self.obs_type == GLYPHS:
            obs = obs[self.obs_type].flatten()
        elif self.obs_type == PIXEL:
            obs = reshape(obs[self.obs_type])
        return obs, reward, done, info
