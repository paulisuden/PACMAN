import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

def epsilon_greedy(state, num):
    if num == 0:
        return actions[np.argmax(q_table[state])]
    else:
        #devuelvo el segundo valor mayor
        q_values = q_table[state]
        sorted_indices = np.argsort(q_values)[::-1]
        return actions[sorted_indices[1]]

episodes = 40
actions = [1, 2, 3, 4]  

reward_list = []
lives_list = []
results = []
pacman_found_list = []

q_table = load_q_table()

small_point_list = []
big_point_list = []
ghosts_list = []
fruit_list = []

for episode in range(episodes):

    print(f"Episode: {episode+1}")
    render = False
    env = make_env(render)
    obs, _ = env.reset()
    done = False
    randomA = False
    total_reward = 0
    steps = 0
    stuck = 0
    state, pacman_found_list = extract_features(obs, pacman_found_list)
    second_action = False

    small_point = 0
    big_point = 0
    ghosts = 0
    fruit = 0

    while not done:

        if second_action: 
            action = epsilon_greedy(state, 1)
            stuck = 0
        else: action = epsilon_greedy(state, 0)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        new_state, pacman_found_list = extract_features(obs, pacman_found_list)

        if reward == 1:
            small_point += 1
        elif reward == 5:
            big_point += 1
        elif reward == 100:
            fruit += 1
        elif reward >= 20:
            ghosts += 1
        
        if state == new_state and state != (1, 3, 3, 1): stuck += 1
        else: stuck = 0
        
        if stuck >= 2: second_action = True
        else: second_action = False

        total_reward += reward
        state = new_state

    small_point_list.append(small_point)
    big_point_list.append(big_point)
    fruit_list.append(fruit)
    ghosts_list.append(ghosts)

    lives = info["lives"]
    if lives > 0: win = 1 
    else: win = 0

    lives_list.append(lives)
    results.append({
        "episode": episode + 1,
        "score": total_reward,
        "win": win
    })
    env.close()


df = pd.DataFrame(results)
#df.to_csv("barplot_q_learning_results_mode2.csv", index=False)

episodes_range = df["episode"]

# Score por episodio 
plt.figure(figsize=(10, 5))
plt.bar(episodes_range, df["score"], color="skyblue", edgecolor="black")
plt.title("Score por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Score")
plt.xticks(episodes_range)
plt.tight_layout()
plt.savefig("code/q-learning/graphics/score.png")

# Frutas y Fantasmas 
plt.figure(figsize=(10, 5))
plt.plot(episodes_range, ghosts_list, label="Fantasmas", color="red", marker="o")
plt.plot(episodes_range, fruit_list, label="Frutas", color="green", marker="o")
plt.axhline(y=sum(ghosts_list)/len(ghosts_list), color="red", linestyle="--", label="Prom. Fantasmas")
plt.axhline(y=sum(fruit_list)/len(fruit_list), color="green", linestyle="--", label="Prom. Frutas")
plt.title("Fantasmas y Frutas por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Cantidad")
plt.legend()
plt.xticks(episodes_range)
plt.tight_layout()
plt.savefig("code/q-learning/graphics/ghosts_and_fruits.png")

# Puntos chicos y grandes
plt.figure(figsize=(10, 5))
plt.plot(episodes_range, small_point_list, label="Puntos chicos", color="blue", marker="o")
plt.plot(episodes_range, big_point_list, label="Puntos grandes", color="orange", marker="o")
plt.axhline(y=sum(small_point_list)/len(small_point_list), color="blue", linestyle="--", label="Prom. chicos")
plt.axhline(y=sum(big_point_list)/len(big_point_list), color="orange", linestyle="--", label="Prom. grandes")
plt.title("Puntos por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Cantidad")
plt.legend()
plt.xticks(episodes_range)
plt.tight_layout()
plt.savefig("code/q-learning/graphics/small_and_big_points.png")