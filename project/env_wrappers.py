import gym
import numpy as np
from collections import deque
import cv2

class RewardClippingWrapper(gym.RewardWrapper):
    def __init__(self, env, min_reward=-1, max_reward=1):
        super(RewardClippingWrapper, self).__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)
    
    def reward2 (self,reward):
        if reward > 0:
            reward = self.max_reward
        elif reward < 0:
            reward = self.min_reward
        return reward

class PreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, observation):
        grayscale_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_observation = cv2.resize(grayscale_observation, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized_observation, axis=-1)

class FrameStackingWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        super(FrameStackingWrapper, self).__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)  # Create a deque with a fixed size of k
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], k), dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs,tuple):    # check if type is tuple, convert to array
            obs = obs[0]
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)  # Fill the deque with the initial observation
        return self._get_observation()

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        if isinstance(obs,tuple):    # check if type is tuple, convert to array
            obs = obs[0]
        self.frames.append(obs)  # Add the new observation, automatically removing the oldest if deque is full
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        stacked_frame = np.stack(self.frames, axis=-1)  # Stack frames along the last dimension
        return stacked_frame
