import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from utils import *
import time

def epsilon_greedy(state, num):
    if num == 0:
        return actions[np.argmax(q_table[state])]
    else:
        #devuelvo el segundo valor mayor
        q_values = q_table[state]
        sorted_indices = np.argsort(q_values)[::-1]
        return actions[sorted_indices[1]]
    
def plotStats(ylabel, title, iter, vector, filename):
    x = list(range(1, iter+1))
    plot.bar(x, vector)
    plot.xlabel("Episodio")
    plot.ylabel(ylabel)
    plot.title(title)
    plot.savefig("code/q-learning/graphics/" + filename + ".png")
    plot.close()

def plotBoxAndWhiskers(vector, ylabel, filename, iter, title):
    x = list(range(1, iter))
    plot.boxplot(vector, vert=True)
    plot.ylabel(ylabel)
    plot.title(title)
    plot.savefig("code/q-learning/graphics/boxplot_" + filename + ".png")
    plot.close()

start = time.time()

SEED = 2025
#random.seed(SEED)
#np.random.seed(SEED)

episodes = 100
actions = [1, 2, 3, 4]  

reward_list = []
lives_list = []
results = []
pacman_found_list = []

q_table = load_q_table()

small_point_list = []
big_point_list = []
ghosts_list = []
steps_list = []
reward_list = []

env = make_env(False)
obs, _ = env.reset(seed=SEED)
for episode in range(episodes):

    print(f"Episode: {episode+1}")
    
    #render = False
    #env = make_env(render)
    #obs, _ = env.reset(seed=SEED + episode)
    obs, _ = env.reset()
    #obs, _ = env.reset(seed=SEED)

    done = False
    total_reward = 0
    stuck = 0
    state, pacman_found_list = extract_features(obs, pacman_found_list)
    second_action = False

    small_point = 0
    big_point = 0
    ghosts = 0
    steps = 0

    while not done:
        steps += 1
        if second_action: 
            action = epsilon_greedy(state, 1)
            stuck = 0
        else: action = epsilon_greedy(state, 0)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        new_state, pacman_found_list = extract_features(obs, pacman_found_list)
        
        if state == new_state and state != (1, 3, 3, 1): stuck += 1
        else: stuck = 0
        
        if stuck >= 2: second_action = True
        else: second_action = False

        state = new_state

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


plotStats("Métrica integradora", "Métrica integradora alcanzada por episodio", episodes, reward_list, "rewards-Q-learning-mode-2")
plotStats("Ghosts", "Fantasmas comidos por episodio", episodes, ghosts_list, "ghosts-Q-learning-mode-2")
plotStats("Points", "Puntos comidos por episodio", episodes, small_point_list, "points-Q-learning-mode-2")
plotStats("Big points", "Puntos grandes comidos por episodio", episodes, big_point_list, "bigPoints-Q-learning-mode-2")
plotStats("Steps", "Pasos dados por episodio", episodes, steps_list, "steps-Q-learning-mode-2")

plotBoxAndWhiskers(reward_list, "Reward", "rewards-Q-learning-mode-2", episodes, "Boxplot de reward por episodio")
plotBoxAndWhiskers(ghosts_list, "Ghosts", "ghosts-Q-learning-mode-2", episodes, "Boxplot de fantasmas comidos por episodio")
plotBoxAndWhiskers(small_point_list, "Points", "points-Q-learning-mode-2", episodes, "Boxplot de puntos comidos por episodio")
plotBoxAndWhiskers(big_point_list, "Big Points", "bigPoints-Q-learning-mode-2", episodes, "Boxplot de puntos grandes comidos por episodio")
plotBoxAndWhiskers(steps_list, "Steps", "steps-Q-learning-mode-2", episodes, "Boxplot de pasos dados por episodio")

os.makedirs("graficos", exist_ok=True)
with open("code/q-learning/csv/means-Q-learning-mode-2.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Métrica", "Promedio"])
    writer.writerow(["Reward", sum(reward_list) / len(reward_list)])
    writer.writerow(["Ghosts", sum(ghosts_list) / len(ghosts_list)])
    writer.writerow(["Points", sum(small_point_list) / len(small_point_list)])
    writer.writerow(["Big Points", sum(big_point_list) / len(big_point_list)])
    writer.writerow(["Steps", sum(steps_list) / len(steps_list)])

with open("code/q-learning/csv/results-Q-learning-mode-2.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episodio", "Reward", "Ghosts", "Points", "Big Points", "Steps"])
    for i in range(len(reward_list)):
        writer.writerow([i + 1, reward_list[i], ghosts_list[i], small_point_list[i], big_point_list[i], steps_list[i]])

end = time.time()
print("Total time: ", end-start)
