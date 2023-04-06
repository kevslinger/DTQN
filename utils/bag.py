from typing import Tuple, Union

import numpy as np


class Bag:
    """A Dataclass dedicated to storing important observations that would have fallen out of the agent's context

    Args:
        bag_size: Size of bag
        obs_mask: The mask to use to indicate the observation is padding
        obs_length: shape of an observation
    """

    def __init__(self, bag_size: int, obs_mask: Union[int, float], obs_length: int):
        self.size = bag_size
        self.obs_mask = obs_mask
        self.obs_length = obs_length
        # Current position in bag
        self.pos = 0

        self.obss, self.actions = self.make_empty_bag()

    def reset(self) -> None:
        self.pos = 0
        self.obss, self.actions = self.make_empty_bag()

    def add(self, obs: np.ndarray, action: int) -> bool:
        if not self.is_full:
            self.obss[self.pos] = obs
            self.actions[self.pos] = action
            self.pos += 1
            return True
        else:
            # Reject adding the observation-action
            return False

    def export(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.obss[: self.pos], self.actions[: self.pos]

    def make_empty_bag(self) -> np.ndarray:
        # Image
        if isinstance(self.obs_length, tuple):
            return np.full((self.size, *self.obs_length), self.obs_mask), np.full(
                (self.size, 1), 0
            )
        # Non-Image
        else:
            return np.full((self.size, self.obs_length), self.obs_mask), np.full(
                (self.size, 1), 0
            )

    @property
    def is_full(self) -> bool:
        return self.pos >= self.size
