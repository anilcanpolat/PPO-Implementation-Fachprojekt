import gymnasium as gym
import numpy as np


class RandomAgent:
    """RandomAgent creates an agent that behaves randomly on a defined action space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return self.action_space.sample(), None
