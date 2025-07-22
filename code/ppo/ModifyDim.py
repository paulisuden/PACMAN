from stable_baselines3.common.preprocessing import preprocess_obs
from gymnasium import ObservationWrapper
import gymnasium as gym
import numpy

class ModifyDim(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        new_shape = (1, *obs_shape)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=numpy.uint8)

    def observation(self, obs):
        return numpy.expand_dims(obs, axis=0)