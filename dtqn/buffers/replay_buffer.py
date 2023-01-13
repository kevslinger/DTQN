import numpy as np
import random
from typing import Optional, Tuple, Union


class ReplayBuffer:
    """
    FIFO Replay Buffer which stores contexts of length ``context_len`` rather than single
        transitions

    Args:
        buffer_size: The number of transitions to store in the replay buffer
        env_obs_length: The size (length) of the environment's observation
        context_len: The number of transitions that will be stored as an agent's context. Default: 1
    """

    def __init__(
        self,
        buffer_size: int,
        env_obs_length: Union[int, Tuple],
        max_episode_steps: int,
        context_len: Optional[int] = 1,
    ):
        self.max_size = buffer_size // max_episode_steps
        self.context_len = context_len
        self.pos = [0, 0]

        # Image domains
        if isinstance(env_obs_length, tuple):
            self.obss = np.zeros(
                [
                    self.max_size,
                    max_episode_steps + 1,
                    *env_obs_length,
                ],
                dtype=np.uint8,
            )
        else:
            self.obss = np.zeros(
                [
                    self.max_size,
                    max_episode_steps + 1,
                    env_obs_length,
                ],
                dtype=np.float32,
            )

        self.actions = np.zeros(
            [self.max_size, max_episode_steps + 1, 1],
            dtype=np.uint8,
        )
        self.rewards = np.zeros(
            [self.max_size, max_episode_steps + 1, 1],
            dtype=np.float32,
        )
        self.dones = np.zeros(
            [self.max_size, max_episode_steps + 1, 1],
            dtype=np.bool_,
        )
        self.episode_lengths = np.zeros([self.max_size], dtype=np.uint8)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_length: Optional[int] = 0,
    ) -> None:
        episode_idx = self.pos[0] % self.max_size
        obs_idx = self.pos[1]
        self.obss[episode_idx, obs_idx] = obs
        self.actions[episode_idx, obs_idx] = action
        self.rewards[episode_idx, obs_idx] = reward
        self.dones[episode_idx, obs_idx] = done
        self.episode_lengths[episode_idx] = episode_length
        self.pos = [self.pos[0], self.pos[1] + 1]

    def can_sample(self, batch_size: int) -> bool:
        return batch_size < self.pos[0]

    def flush(self):
        self.pos = [self.pos[0] + 1, 0]

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        episode_idxes = np.array(
            [
                [random.randint(0, min(self.pos[0], self.max_size) - 1)]
                for _ in range(batch_size)
            ]
        )
        transition_starts = np.array(
            [
                random.randint(
                    0, max(0, self.episode_lengths[idx[0]] - self.context_len)
                )
                for idx in episode_idxes
            ]
        )
        transitions = np.array(
            [range(start, start + self.context_len) for start in transition_starts]
        )
        return (
            self.obss[episode_idxes, transitions],
            self.actions[episode_idxes, transitions],
            self.rewards[episode_idxes, transitions],
            self.obss[episode_idxes, 1 + transitions],
            self.dones[episode_idxes, transitions],
            self.episode_lengths[episode_idxes],
        )
