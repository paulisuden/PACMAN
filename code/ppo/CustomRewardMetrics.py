import gymnasium as gym
from gymnasium import Wrapper

class CustomRewardMetrics(Wrapper):
    def __init__(self, environment):
            super().__init__(environment)
            self.lives = None
            self.stepsWithoutReward = 0

    def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.lives = 4
            return obs, info

    def step(self, executedAction):
        obs, reward, terminated, truncated, info = self.env.step(executedAction)
        #print(info)
        livesNew = info["lives"]

        
        # detecto si las vidas actuales son diferentes a las almacenadas
        if reward > 0:
            if reward == 20 or reward == 40 or reward == 80 or reward == 160:
                reward = 0
            elif reward == 100:
                reward = 0
            else:
                reward = 1

        return obs, reward, terminated, truncated, info