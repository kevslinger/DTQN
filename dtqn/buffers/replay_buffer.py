import numpy as np
import random
from typing import Optional, Tuple, Union

from utils.env_processing import Bag


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
        obs_mask: int,
        max_episode_steps: int,
        context_len: Optional[int] = 1,
    ):
        self.max_size = buffer_size // max_episode_steps
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        self.max_episode_steps = max_episode_steps
        self.obs_mask = obs_mask
        self.pos = [0, 0]

        # Image domains
        if isinstance(env_obs_length, tuple):
            self.obss = np.full(
                [
                    self.max_size,
                    max_episode_steps + 1,  # Keeps first and last obs together for +1
                    *env_obs_length,
                ],
                obs_mask,
                dtype=np.uint8,
            )
        else:
            self.obss = np.full(
                [
                    self.max_size,
                    max_episode_steps + 1,  # Keeps first and last obs together for +1
                    env_obs_length,
                ],
                obs_mask,
                dtype=np.float32,
            )

        self.actions = np.zeros(
            [self.max_size, max_episode_steps, 1],
            dtype=np.uint8,
        )
        self.rewards = np.zeros(
            [self.max_size, max_episode_steps, 1],
            dtype=np.float32,
        )
        self.dones = np.ones(
            [self.max_size, max_episode_steps, 1],
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
        self.obss[episode_idx, obs_idx + 1] = obs
        self.actions[episode_idx, obs_idx] = action
        self.rewards[episode_idx, obs_idx] = reward
        self.dones[episode_idx, obs_idx] = done
        self.episode_lengths[episode_idx] = episode_length
        self.pos = [self.pos[0], self.pos[1] + 1]

    def store_obs(self, obs: np.ndarray) -> None:
        """Use this at the beginning of the episode to store the first obs"""
        episode_idx = self.pos[0] % self.max_size
        self.cleanse_episode(episode_idx)
        self.obss[episode_idx, 0] = obs

    def can_sample(self, batch_size: int) -> bool:
        return batch_size < self.pos[0]

    def flush(self):
        self.pos = [self.pos[0] + 1, 0]

    def cleanse_episode(self, episode_idx: int) -> None:
        # Cleanse the episode of any previous data
        # Image domains
        if isinstance(self.env_obs_length, tuple):
            self.obss[episode_idx] = np.full(
                [
                    self.max_episode_steps
                    + 1,  # Keeps first and last obs together for +1
                    *self.env_obs_length,
                ],
                self.obs_mask,
                dtype=np.uint8,
            )
        else:
            self.obss[episode_idx] = np.full(
                [
                    self.max_episode_steps
                    + 1,  # Keeps first and last obs together for +1
                    self.env_obs_length,
                ],
                self.obs_mask,
                dtype=np.float32,
            )
        self.actions[episode_idx] = np.zeros(
            [self.max_episode_steps, 1],
            dtype=np.uint8,
        )
        self.rewards[episode_idx] = np.zeros(
            [self.max_episode_steps, 1],
            dtype=np.float32,
        )
        self.dones[episode_idx] = np.ones(
            [self.max_episode_steps, 1],
            dtype=np.bool_,
        )
        self.episode_lengths[episode_idx] = 0

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Exclude the current episode we're in
        valid_episodes = [
            i
            for i in range(min(self.pos[0], self.max_size))
            if i != self.pos[0] % self.max_size
        ]
        episode_idxes = np.array(
            [[random.choice(valid_episodes)] for _ in range(batch_size)]
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

    # TODO:
    def sample_with_bag(
        self, batch_size: int, sample_bag: Bag
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        episode_idxes = np.array(
            [
                # Exclude the current episode we're in
                [
                    random.choice(
                        [
                            i
                            for i in range(min(self.pos[0], self.max_size))
                            if i != self.pos[0]
                        ]
                    )
                ]
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

        # Create `batch_size` replica bags
        bags = np.full(
            [batch_size, sample_bag.bag_size, sample_bag.obs_length],
            sample_bag.obs_mask,
        )

        # Sample from the bag with observations that won't be in the main context
        for bag_idx in range(batch_size):
            # Possible bag is smaller than max bag size, so take all of it
            if transition_starts[bag_idx] < sample_bag.bag_size:
                bags[bag_idx, : transition_starts[bag_idx]] = self.obss[
                    episode_idxes[bag_idx], : transition_starts[bag_idx]
                ]
            # Otherwise, randomly sample
            else:
                bags[bag_idx] = np.array(
                    random.sample(
                        self.obss[episode_idxes[bag_idx], : transition_starts[bag_idx]]
                        .squeeze()
                        .tolist(),
                        k=sample_bag.bag_size,
                    )
                )
        return (
            self.obss[episode_idxes, transitions],
            self.actions[episode_idxes, transitions],
            self.rewards[episode_idxes, transitions],
            self.obss[episode_idxes, 1 + transitions],
            self.dones[episode_idxes, transitions],
            self.episode_lengths[episode_idxes],
            bags,
        )
