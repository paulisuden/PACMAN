import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plot
import time
import pandas as pd
import time
import csv


def plotStats(ylabel, title, iter, vector, filename):
    x = list(range(1, iter+1))
    plot.bar(x, vector)
    plot.xlabel("Episodio")
    plot.ylabel(ylabel)
    plot.title(title)
    plot.savefig("code/random/graphics/" + filename + ".png")
    plot.close()

def plotBoxAndWhiskers(vector, ylabel, filename, iter, title):
    x = list(range(1, iter))
    plot.boxplot(vector, vert=True)
    plot.ylabel(ylabel)
    plot.title(title)
    plot.savefig("code/random/graphics/boxplot_" + filename + ".png")
    plot.close()

start = time.time()

SEED = 2025

episodes = 100
ACTIONS = [1, 2, 3, 4]
results = []

small_point_list = []
big_point_list = []
ghosts_list = []
steps_list = []
reward_list = []

env = gym.make("ALE/Pacman-v5", mode=2)
obs, _ = env.reset(seed=SEED)
for episode in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    small_point = 0
    big_point = 0
    ghosts = 0
    steps = 0

    while not done:
        action = np.random.choice(ACTIONS)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if reward > 0:
                #aumento recompensa por fantasmas
                if any(x <= reward <= x+5 for x in [20, 40, 80, 160]):
                    ghosts += 1
                    if reward in [20, 40, 80, 160]:
                        pass
                    elif reward in [25, 45, 85, 165]:
                        big_point += 1
                    else:
                        small_point += 1
                #aumento recompensa por puntos chicos
                elif reward < 5:
                    small_point += reward
                #aumento recompensa por puntos grandes
                elif reward == 5:
                    big_point += 1
                #aumento recompensa por fruta
                elif reward == 100:
                    pass
                elif reward == 101:
                    small_point += 1
                elif reward == 102:
                    small_point += 2 

    total_reward += ghosts * 5 + small_point + big_point * 3
    small_point_list.append(small_point)
    big_point_list.append(big_point)
    ghosts_list.append(ghosts)
    steps_list.append(steps)
    reward_list.append(total_reward)

env.close()

plotStats("Métrica integradora", "Métrica integradora alcanzada por episodio", episodes, reward_list, "rewards-random-mode-2")
plotStats("Ghosts", "Fantasmas comidos por episodio", episodes, ghosts_list, "ghosts-random-mode-2")
plotStats("Points", "Puntos comidos por episodio", episodes, small_point_list, "points-random-mode-2")
plotStats("Big points", "Puntos grandes comidos por episodio", episodes, big_point_list, "bigPoints-random-mode-2")
plotStats("Steps", "Pasos dados por episodio", episodes, steps_list, "steps-random-mode-2")

plotBoxAndWhiskers(reward_list, "Reward", "rewards-random-mode-2", episodes, "Boxplot de reward por episodio")
plotBoxAndWhiskers(ghosts_list, "Ghosts", "ghosts-random-mode-2", episodes, "Boxplot de fantasmas comidos por episodio")
plotBoxAndWhiskers(small_point_list, "Points", "points-random-mode-2", episodes, "Boxplot de puntos comidos por episodio")
plotBoxAndWhiskers(big_point_list, "Big Points", "bigPoints-random-mode-2", episodes, "Boxplot de puntos grandes comidos por episodio")
plotBoxAndWhiskers(steps_list, "Steps", "steps-random-mode-2", episodes, "Boxplot de pasos dados por episodio")

with open("code/random/csv/means-random-mode-2.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Métrica", "Promedio"])
    writer.writerow(["Reward", sum(reward_list) / len(reward_list)])
    writer.writerow(["Ghosts", sum(ghosts_list) / len(ghosts_list)])
    writer.writerow(["Points", sum(small_point_list) / len(small_point_list)])
    writer.writerow(["Big Points", sum(big_point_list) / len(big_point_list)])
    writer.writerow(["Steps", sum(steps_list) / len(steps_list)])

with open("code/random/csv/results-random-mode-2.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episodio", "Reward", "Ghosts", "Points", "Big Points", "Steps"])
    for i in range(len(reward_list)):
        writer.writerow([i + 1, reward_list[i], ghosts_list[i], small_point_list[i], big_point_list[i], steps_list[i]])

end = time.time()
print("Total time: ", end-start)


########################


results = []

small_point_list = []
big_point_list = []
ghosts_list = []
steps_list = []
reward_list = []

env = gym.make("ALE/Pacman-v5", mode=5)
obs, _ = env.reset(seed=SEED)
for episode in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    small_point = 0
    big_point = 0
    ghosts = 0
    steps = 0

    while not done:
        action = np.random.choice(ACTIONS)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    if reward > 0:
        if any(x <= reward <= x+5 for x in [20, 40, 80, 160]):
            ghosts += 1
            if reward in [20, 40, 80, 160]:
                pass
            elif reward in [25, 45, 85, 165]:
                big_point += 1
            else:
                small_point += 1
        elif reward < 5:
            small_point += reward
        elif reward == 5:
            big_point += 1
        elif reward == 100:
            pass
        elif reward == 101:
            small_point += 1
        elif reward == 102:
            small_point += 2 

    total_reward += ghosts * 5 + small_point + big_point * 3
    small_point_list.append(small_point)
    big_point_list.append(big_point)
    ghosts_list.append(ghosts)
    steps_list.append(steps)
    reward_list.append(total_reward)

env.close()

plotStats("Métrica integradora", "Métrica integradora alcanzada por episodio", episodes, reward_list, "rewards-random-mode-5")
plotStats("Ghosts", "Fantasmas comidos por episodio", episodes, ghosts_list, "ghosts-random-mode-5")
plotStats("Points", "Puntos comidos por episodio", episodes, small_point_list, "points-random-mode-5")
plotStats("Big points", "Puntos grandes comidos por episodio", episodes, big_point_list, "bigPoints-random-mode-5")
plotStats("Steps", "Pasos dados por episodio", episodes, steps_list, "steps-random-mode-5")

plotBoxAndWhiskers(reward_list, "Reward", "rewards-random-mode-5", episodes, "Boxplot de reward por episodio")
plotBoxAndWhiskers(ghosts_list, "Ghosts", "ghosts-random-mode-5", episodes, "Boxplot de fantasmas comidos por episodio")
plotBoxAndWhiskers(small_point_list, "Points", "points-random-mode-5", episodes, "Boxplot de puntos comidos por episodio")
plotBoxAndWhiskers(big_point_list, "Big Points", "bigPoints-random-mode-5", episodes, "Boxplot de puntos grandes comidos por episodio")
plotBoxAndWhiskers(steps_list, "Steps", "steps-random-mode-5", episodes, "Boxplot de pasos dados por episodio")

with open("code/random/csv/means-random-mode-5.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Métrica", "Promedio"])
    writer.writerow(["Reward", sum(reward_list) / len(reward_list)])
    writer.writerow(["Ghosts", sum(ghosts_list) / len(ghosts_list)])
    writer.writerow(["Points", sum(small_point_list) / len(small_point_list)])
    writer.writerow(["Big Points", sum(big_point_list) / len(big_point_list)])
    writer.writerow(["Steps", sum(steps_list) / len(steps_list)])

with open("code/random/csv/results-random-mode-5.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episodio", "Reward", "Ghosts", "Points", "Big Points", "Steps"])
    for i in range(len(reward_list)):
        writer.writerow([i + 1, reward_list[i], ghosts_list[i], small_point_list[i], big_point_list[i], steps_list[i]])

end = time.time()
print("Total time: ", end-start)



################################



results = []

small_point_list = []
big_point_list = []
ghosts_list = []
steps_list = []
reward_list = []

env = gym.make("ALE/Pacman-v5")
obs, _ = env.reset(seed=SEED)
for episode in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    small_point = 0
    big_point = 0
    ghosts = 0
    steps = 0

    while not done:
        action = np.random.choice(ACTIONS)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if reward > 0:
            if any(x <= reward <= x+5 for x in [20, 40, 80, 160]):
                ghosts += 1
                if reward in [20, 40, 80, 160]:
                    pass
                elif reward in [25, 45, 85, 165]:
                    big_point += 1
                else:
                    small_point += 1
            elif reward < 5:
                small_point += reward
            elif reward == 5:
                big_point += 1
            elif reward == 100:
                pass
            elif reward == 101:
                small_point += 1
            elif reward == 102:
                small_point += 2 

    total_reward += ghosts * 5 + small_point + big_point * 3
    small_point_list.append(small_point)
    big_point_list.append(big_point)
    ghosts_list.append(ghosts)
    steps_list.append(steps)
    reward_list.append(total_reward)

env.close()

plotStats("Métrica integradora", "Métrica integradora alcanzada por episodio", episodes, reward_list, "rewards-random-mode-0")
plotStats("Ghosts", "Fantasmas comidos por episodio", episodes, ghosts_list, "ghosts-random-mode-0")
plotStats("Points", "Puntos comidos por episodio", episodes, small_point_list, "points-random-mode-0")
plotStats("Big points", "Puntos grandes comidos por episodio", episodes, big_point_list, "bigPoints-random-mode-0")
plotStats("Steps", "Pasos dados por episodio", episodes, steps_list, "steps-random-mode-0")

plotBoxAndWhiskers(reward_list, "Reward", "rewards-random-mode-0", episodes, "Boxplot de reward por episodio")
plotBoxAndWhiskers(ghosts_list, "Ghosts", "ghosts-random-mode-0", episodes, "Boxplot de fantasmas comidos por episodio")
plotBoxAndWhiskers(small_point_list, "Points", "points-random-mode-0", episodes, "Boxplot de puntos comidos por episodio")
plotBoxAndWhiskers(big_point_list, "Big Points", "bigPoints-random-mode-0", episodes, "Boxplot de puntos grandes comidos por episodio")
plotBoxAndWhiskers(steps_list, "Steps", "steps-random-mode-0", episodes, "Boxplot de pasos dados por episodio")

with open("code/random/csv/means-random-mode-0.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Métrica", "Promedio"])
    writer.writerow(["Reward", sum(reward_list) / len(reward_list)])
    writer.writerow(["Ghosts", sum(ghosts_list) / len(ghosts_list)])
    writer.writerow(["Points", sum(small_point_list) / len(small_point_list)])
    writer.writerow(["Big Points", sum(big_point_list) / len(big_point_list)])
    writer.writerow(["Steps", sum(steps_list) / len(steps_list)])

with open("code/random/csv/results-random-mode-0.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episodio", "Reward", "Ghosts", "Points", "Big Points", "Steps"])
    for i in range(len(reward_list)):
        writer.writerow([i + 1, reward_list[i], ghosts_list[i], small_point_list[i], big_point_list[i], steps_list[i]])

end = time.time()
print("Total time: ", end-start)
