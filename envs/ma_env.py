from abc import ABC
from gym import Env, Space


class MultiAgentEnv(Env, ABC):
    def __init__(self):
        self.n_agents = 0
        self.observation_space = Space()
        self.action_space = Space()
