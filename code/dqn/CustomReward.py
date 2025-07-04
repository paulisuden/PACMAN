import gymnasium as gym
from gymnasium import Wrapper

class CustomReward(Wrapper):
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
        if reward > 0:
            reward = reward * 0.8
            #aumento recompensa por puntos chicos
            if reward < 5:
                reward = reward * 1
            #aumento recompensa por puntos grandes
            elif reward == 5:
                reward = reward * 1
            #aumento recompensa por fruta
            elif reward == 100:
                reward = 0
            #aumento recompensa si hubiese algun reward mayor a 161
            elif reward > 161:
                reward = reward * 1
            #aumento recompensa por comer fantasmas
            else:
                reward = reward * 1
        if livesNew < self.lives:
            reward -= 50
            self.lives = livesNew
            

        return obs, reward, terminated, truncated, info