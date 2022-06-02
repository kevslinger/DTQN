import numpy as np
import random
from typing import Optional, Tuple


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
        self, buffer_size: int, env_obs_length: int, context_len: Optional[int] = 1
    ):
        self.max_size = buffer_size
        self.context_len = context_len
        self.pos = 0

        self.obss = np.zeros(
            [buffer_size, self.context_len, env_obs_length], dtype=np.float32
        )
        self.next_obss = np.zeros(
            [buffer_size, self.context_len, env_obs_length], dtype=np.float32
        )
        self.actions = np.zeros([buffer_size, self.context_len, 1], dtype=np.uint8)
        self.rewards = np.zeros([buffer_size, self.context_len, 1], dtype=np.float32)
        self.dones = np.zeros([buffer_size, self.context_len, 1], dtype=np.bool_)
        self.episode_lengths = np.zeros([buffer_size], dtype=np.uint8)

    def store(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_length: Optional[int] = 0,
    ) -> None:
        idx = self.pos % self.max_size
        self.obss[idx] = obs
        self.next_obss[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.episode_lengths[idx] = episode_length
        self.pos += 1

    def can_sample(self, batch_size: int) -> bool:
        return batch_size <= self.pos

    def full(self) -> bool:
        return self.pos >= self.max_size

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxes = [
            random.randint(0, min(self.pos, self.max_size) - 1)
            for _ in range(batch_size)
        ]
        return (
            self.obss[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.next_obss[idxes],
            self.dones[idxes],
            self.episode_lengths[idxes],
        )
