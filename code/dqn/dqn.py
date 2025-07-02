import gymnasium as gym
from stable_baselines3 import DQN
from GreyScaleObs import GreyScaleObs
from gymnasium.wrappers import ResizeObservation, FrameStack
from datetime import datetime
import numpy
import torch
print(torch.version.hip) 
print(torch.cuda.is_available()) 
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def applyWrappers(env):
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

#Train con DQN 
env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode="rgb_array"))

model = DQN("CnnPolicy", env = env, buffer_size= 200000, batch_size = 32, verbose=1, exploration_fraction=0.15, tensorboard_log="./tensorboardDQNPacman/" + date, device="cuda")

model.learn(total_timesteps=2000)
model.save("pacmanDqn" + date)


#Test
env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode="human"))
model = DQN.load("./pacmanDqn2m.zip", env=env)

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