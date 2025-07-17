import gymnasium as gym
from gymnasium import Wrapper

class CustomRewardDefinitiveMetrics(Wrapper):
    def __init__(self, environment):
            super().__init__(environment)
            self.lives = None
            self.stepsWithoutReward = 0
            self.points = 0
            self.ghosts = 0
            self.steps = 0
            self.bigPoint = 0

    def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.lives = 4
            self.points = 0
            self.ghosts = 0
            self.steps = 0
            self.bigPoint = 0
            return obs, info

    def step(self, executedAction):
        obs, reward, terminated, truncated, info = self.env.step(executedAction)
        #print(info)
        livesNew = info["lives"]

        
        if reward > 0:
            #aumento recompensa por fantasmas
            if any(x <= reward <= x+5 for x in [20, 40, 80, 160]):
                self.ghosts += 1
                if reward in [20, 40, 80, 160]:
                    pass
                elif reward in [25, 45, 85, 165]:
                    self.bigPoint += 1
                else:
                    self.points += 1
            #aumento recompensa por puntos chicos
            elif reward < 5:
                self.points += reward
            #aumento recompensa por puntos grandes
            elif reward == 5:
                self.bigPoint += 1
            #aumento recompensa por fruta
            elif reward == 100:
                pass
            elif reward == 101:
                self.points += 1
            elif reward == 102:
                self.points += 2 
            #aumento recompensa por algún bonus
            else:
                print(reward)
                pass
        if livesNew < self.lives:
            self.lives = livesNew
        self.steps += 1
        info.update({
            "totalSteps": self.steps,
            "points": self.points,
            "bigPoint": self.bigPoint,
            "ghosts": self.ghosts
        })
        return obs, reward, terminated, truncated, info