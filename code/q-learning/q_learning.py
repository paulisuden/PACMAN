import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import time
from utils import *

def epsilon_greedy(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(q_table[state])]
    
"""SEED = 2025
random.seed(SEED)
np.random.seed(SEED)"""

start = time.time()
learning_rate = 0.1
discount_factor = 0.95

epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01

episodes = 2000
actions = [1, 2, 3, 4]  
reward_list = []
penalty_list = []

q_table = load_q_table()

pacman_found_list = []

for episode in range(episodes):

    print(f"Episode: {episode+1}")

    render = False

    env = make_env(render)
    obs, _ = env.reset()
    #obs, _ = env.reset(seed=SEED)
    state = extract_features(obs)
    done = False

    randomA = False
    total_reward = 0
    last_state = None
    last_action = None
    stuck = 0

    while not done:
        
        if randomA == False:
            action = epsilon_greedy(state)
        else:
            action = random.choice(actions)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = extract_features(next_obs)
        
        # stuck
        if last_state is not None and state == last_state: 
            randomA = True
            if action == last_action:
                stuck = -3
            else:
                stuck = 0
        else:
            randomA = False
            stuck = 0

        total_reward += reward

        # penalties
        if state[action - 1] == 0:   
            penalty = -10 #ghost
        elif state[action - 1] == 1:
            penalty = -0.5 #pared
        elif state[action-1] == 2:
            penalty = 3 #libre
        else:
            penalty = 15 #pellet

        reward_q_table = reward + penalty + stuck

        # actualizar q-table
        best_next = float(np.max(q_table[next_state]))
        q_table[state][action - 1] = round(
        q_table[state][action - 1] + learning_rate * (reward_q_table + discount_factor * best_next - 
                                                                    q_table[state][action - 1]), 3)

        last_state = state
        state = next_state
        last_action = action
        obs = next_obs  

        penalty_list.append(reward_q_table)

    env.close()
    reward_list.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(reward_list[-10:])
        avg_penalty = np.mean(penalty_list[-10:])
        print(f"→ Episodio {episode+1} | Promedio: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
        print("Penalty mean: ", avg_penalty )


# guardar q table 
with open('q_table.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['state', 'action_1', 'action_2', 'action_3', 'action_4'])  
    
    for state, action_values in q_table.items():
        writer.writerow([state] + list(action_values))

# graficar
plt.figure(figsize=(10, 5))
plt.plot(reward_list, label="Recompensa por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("Evolución de la recompensa durante el entrenamiento")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("histogram_q_learning_agent.png")

end = time.time()
total_time = end - start
print("pacman found: ", np.mean(pacman_found_list))
print("total time: ", total_time)