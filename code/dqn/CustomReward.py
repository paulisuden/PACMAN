import gymnasium as gym
from gymnasium import Wrapper

class CustomReward(Wrapper):
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
            #aumento recompensa por fantasmas
            if any(x <= reward <= x+4 for x in [20, 40, 80, 160]):
                reward = 1
            #aumento recompensa por puntos chicos
            elif reward < 5:
                reward = 0.3
            #aumento recompensa por puntos grandes
            elif reward == 5:
                reward = 0.4
            #aumento recompensa por fruta
            elif reward == 100:
                reward = 0
            #aumento recompensa por algún bonus
            else:
                print(reward)
                reward = 0
        elif reward == 0:
            self.stepsWithoutReward += 1
            if self.stepsWithoutReward >= 5:
                reward = -0.05
                self.stepsWithoutReward = 0
        if livesNew < self.lives:
            reward -= 1
            self.lives = livesNew

        return obs, reward, terminated, truncated, info
