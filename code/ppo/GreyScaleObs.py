import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy
import cv2

class GreyScaleObs(ObservationWrapper):
    def __init__(self, baseEnvironment):
        super().__init__(baseEnvironment)
        #se guarda el alto y el ancho
        sizeObs = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=sizeObs, dtype=numpy.uint8)

    #Recibo una imagen (observation)
    def observation(self, observation):
        #Devuelvo la misma imagen pero en escala de grises
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)