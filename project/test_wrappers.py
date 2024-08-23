import gym
import matplotlib.pyplot as plt
import numpy as np
from env_wrappers import RewardClippingWrapper, FrameStackingWrapper, PreprocessingWrapper

class TestWrappers:
    def __init__(self, env_name='FlappyBird-pixels-v0'): #tests were made with own FlappyBirdenv
        self.env = gym.make(env_name)
        self.env = RewardClippingWrapper(self.env)
        self.env = PreprocessingWrapper(self.env) 
        self.env = FrameStackingWrapper(self.env)

    def test_reset(self):
        obs = self.env.reset()
        self._show_observation(obs, 'Initial Observation After Reset')
        return obs

    def test_step(self, action):
        obs, reward, done, _,info = self.env.step(action)
        self._show_observation(obs, f'Observation After Step Action {action}')
        print(f'Reward: {reward}, Done: {done}, Info: {info}')
        return obs, reward, done, info

    def _show_observation(self, obs, title):
        if isinstance(obs, np.ndarray):
            print(f"Observation is a numpy array. Shape: {obs.shape}")
            plt.figure()
            plt.imshow(obs[:, :, 0], cmap='gray')
            plt.title(title)
            plt.show()
        else:
            print(f'Observation type {type(obs)} is not supported for visualization.')

# Usage
if __name__ == "__main__":
    tester = TestWrappers()
    tester.test_reset()
    tester.test_step(0)  # Example action
