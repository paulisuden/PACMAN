import gymnasium as gym
from stable_baselines3 import DQN
from GreyScaleObs import GreyScaleObs
from CustomReward import CustomReward
import matplotlib.pyplot as plot
from gymnasium.wrappers import ResizeObservation, FrameStack
from datetime import datetime
import numpy
import torch
import os
from CustomRewardMetrics import CustomRewardMetrics

print(torch.version.hip) 
print(torch.cuda.is_available()) 
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def applyWrappers(env, option="train"):
    if option == "train":
        env = CustomReward(env)
    else:
        env = CustomRewardMetrics(env)
    env = GreyScaleObs(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, num_stack=4)
    return env

"""env = gym.make("ALE/Pacman-v5", frameskip=4, render_mode="human")
while True:
    
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
"""

"""
#Train con DQN 
env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode="rgb_array"))

model = DQN("CnnPolicy", env = env, buffer_size= 200000, batch_size = 32, verbose=1, exploration_fraction=0.15, tensorboard_log="./tensorboardDQNPacman/" + date, device="cuda")

model.learn(total_timesteps=2000000)
model.save("pacmanDqn" + date)
"""

"""
#Test
print("Directorio actual:", os.getcwd())
env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode="human"))
model = DQN.load("pacmanDqn2MNewReward2", env=env)

obs, _ = env.reset(seed=2025)
done = False

while not done:
    action, _ = model.predict(numpy.array(obs), deterministic=True)
    #print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    #if (reward != 0 and reward != 1 and reward != 2):
    #    print(reward)
    #    input()
    done = terminated or truncated
env.close()
"""


#Para extraer métricas, NO se renderiza
iter = 101
env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode=None), option="test")
model = DQN.load("/home/tomas/Desktop/PacmanRL/DQN/pacmanDqn2MNewReward3.zip", env=env)
rewardsVector = []
obs, _ = env.reset(seed=2025)

for i in range(1, iter):
    rewardsSum = 0
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(numpy.array(obs), deterministic=True)
        #print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewardsSum += reward
        #if (reward != 0 and reward != 1 and reward != 2):
        #    print(reward)
        #    input()
        done = terminated or truncated
    if info["lives"] > 0:
        print("Winner") 
    rewardsVector.append(rewardsSum)
env.close()

sum = 0
for i in rewardsVector:
    sum += i
sum = sum / len(rewardsVector)
print(sum)
x = list(range(1, iter))
plot.bar(x, rewardsVector)
plot.xlabel("Episodio")
plot.ylabel("Reward alcanzada")
plot.title("Reward alcanzado por episodio")
os.makedirs("graficos", exist_ok=True)
plot.savefig("graficos/rewards2MNew3.png")
