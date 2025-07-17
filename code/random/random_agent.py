import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make("ALE/Pacman-v5") 

NUM_EPISODES = 10
ACTIONS = [1, 2, 3, 4]
results = []

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = np.random.choice(ACTIONS)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    results.append({
        "episode": episode + 1,
        "score": total_reward,
        "time_alive": steps,
    })

    print(f"Episode {episode + 1} | Score: {total_reward:.0f} | Time alive: {steps}")

env.close()


"""df = pd.DataFrame(results)
df.to_csv("code/random/random_agent_results.csv", index=False)


plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.hist(df["score"], bins=10, color="skyblue", edgecolor="black")
plt.title("Score")
plt.xlabel("Score")
plt.ylabel("Episodes")

plt.subplot(1, 3, 2)
plt.hist(df["time_alive"], bins=10, color="orange", edgecolor="black")
plt.title("Time Alive")
plt.xlabel("Steps")
plt.ylabel("Episodes")

plt.tight_layout()
plt.savefig("code/random/graphics")  
plt.show()
"""
