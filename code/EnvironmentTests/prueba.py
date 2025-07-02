import gymnasium as gym
from ale_py import ALEInterface

"""
env = gym.make("ALE/MsPacman-v5", render_mode="human")

while True:
    obs, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(truncated)
        print(info)

        #Los puntos chicos dan 10
        #if (reward != 0.0 and reward != 10.0):
        #   input()
        #Los puntos grandes dan 50
        if (reward > 50):
            input()
    env.close()
"""

env = gym.make("ALE/Pacman-v5", render_mode="human")

while True:
    obs, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(truncated)
        print(info)

        #Los puntos chicos dan 1
        #if (reward != 1 and reward != 0):
        #   input()
        #Los puntos grandes dan 5
        if (reward > 5):
            input()
    env.close()