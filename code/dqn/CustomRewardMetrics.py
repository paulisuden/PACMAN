import gymnasium as gym
from gymnasium import Wrapper

class CustomRewardMetrics(Wrapper):
    def __init__(self, environment):
            super().__init__(environment)
            self.lives = None

    def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.lives = 4
            return obs, info

    def step(self, executedAction):
        obs, reward, terminated, truncated, info = self.env.step(executedAction)
        #print(info)
        livesNew = info["lives"]

        # detecto si las vidas actuales son diferentes a las almacenadas
        if livesNew < self.lives:
            reward -= 25
            self.lives = livesNew
        else:
            # aumento la recompensa por los puntos chicos
            if reward == 2 or reward == 1:
                reward = reward * 3
            # aumento recompensa por puntos grandes
            if reward == 5:
                reward = reward * 1
            # aumento recompensa por matar fantasmas
            if reward == 20 or reward == 40 or reward == 80 or reward == 160:
                reward = reward * 1
            

        return obs, reward, terminated, truncated, info