import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import ResizeObservation
import matplotlib.pyplot as plt
import csv
import os


def load_q_table(filename="q_table.csv"):
    q_table = defaultdict(lambda: np.zeros(4))
    if os.path.exists(filename):
        with open(filename, mode='r') as f:
            reader = csv.reader(f)
            next(reader)  # jump header
            for row in reader:
                state = eval(row[0])  # convert a string "(1, 0, 0, 1)" to a tuple
                values = list(map(float, row[1:]))
                q_table[state] = np.array(values)
    return q_table

######################################################## CROP AND RESIZE OBSERVATION ########################################################

class CropObservation(ObservationWrapper):
    def __init__(self, env, top=0, bottom=0, left=0, right=0):
        super().__init__(env)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

        obs_shape = self.observation_space.shape
        if len(obs_shape) == 3:
            h, w, c = obs_shape
            new_shape = (h - top - bottom, w - left - right, c)
        elif len(obs_shape) == 2:
            h, w = obs_shape
            new_shape = (h - top - bottom, w - left - right)
        else:
            raise ValueError("Error in observation shape:", obs_shape)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8
        )

    def observation(self, obs):
        return obs[self.top:obs.shape[0] - self.bottom,
                    self.left:obs.shape[1] - self.right]

######################################################## MAKE ENVIRONMENT ########################################################

def make_env(apply_wrappers=True):
    env = gym.make("ALE/Pacman-v5", difficulty=1)
    if apply_wrappers:
        env = CropObservation(env, bottom=38)           # crop 
        env = ResizeObservation(env, shape=(80,80))     # resize 
    return env

######################################################## EXTRACT FEATURES ########################################################

# to ckeck the crop
def show_local_view(local_view):
    plt.imshow(local_view)
    plt.title("Local View centrada en Pac-Man")
    plt.axis('off') 
    plt.show()

GHOST_COLORS = {
    #'red': np.array([200, 72, 72]),
    'pink': np.array([252, 144, 200]),
    #'blue': np.array([84, 138, 209]),
    #'orange': np.array([198, 151, 52]),
    #'scared': np.array([132, 210, 222])
}

PELLET_COLORS = {
    '1': np.array([161, 141, 134]), 
    '2': np.array([122, 109, 149]), 
    '3': np.array([181, 157, 127]), 
    '4': np.array([174, 152, 129])
}

PACMAN_COLORS = [
    [233, 208, 147],
    [157, 142, 159],
    [191, 171, 154],
    [252, 224, 144],
    [90, 84, 170],
    [206, 183, 139],
    [147, 134, 158],
    [239, 213, 146],
    [214, 191, 150],
    [189, 170, 154],
    [113, 104, 166],
    [169, 150, 148],
    [131, 118, 154],
]
PACMAN_COLORS = [np.array(c) for c in PACMAN_COLORS]

WALL_COLORS = [
    np.array([223, 192, 111]),
    np.array([145, 128, 140])
]

EXCLUDED_COLORS = PACMAN_COLORS + WALL_COLORS

def count_color_excluding_multiple(zone, color_rgb, tolerance, excluded_colors, exclude_tol):
    # es true en cada píxel que está dentro de la tolerancia respecto al color_rgb
    match = np.all(np.abs(zone - color_rgb) <= tolerance, axis=-1)

    # quitamos las coincidencias con colores de pacman y la pared
    for excl in excluded_colors:
        excl_match = np.all(np.abs(zone - excl) <= exclude_tol, axis=-1)
        match = match & ~excl_match

    return np.sum(match)

def count_color(zone, color_rgb, tolerance):
    return np.sum(np.all(np.abs(zone - color_rgb) <= tolerance, axis=-1))

def count_all_ghosts(zone):
    total = 0
    for color in GHOST_COLORS.values():
        total += count_color(zone, color,20)
    return total

def count_pellets(zone):
    total = 0
    for color in PELLET_COLORS.values():
        total += count_color_excluding_multiple(
            zone,
            color_rgb=color,
            tolerance=10,
            excluded_colors=EXCLUDED_COLORS, 
            exclude_tol=15
        )
    return total

def extract_features(obs):
    h, w, _ = obs.shape
    PACMAN_MAIN_COLOR = np.array([252, 224, 144])
    mask_pacman = np.all(np.abs(obs - PACMAN_MAIN_COLOR) <= 10, axis=-1)
    pos = np.argwhere(mask_pacman)
    
    pacman_found = 0
    if len(pos) == 0:
        y, x = h // 2, w // 2
    else:
        pacman_found = 1
        y, x = pos[0]

    pacman_found_list.append(pacman_found) #para saber cuantas veces se encontró

    # centered crop
    window_size = 29
    half = window_size // 2
    top = max(0, y - half)
    bottom = min(h, y + half + 1)
    left = max(0, x - half)
    right = min(w, x + half + 1)
    local_view = obs[top:bottom, left:right, :]

    center_y = y - top
    center_x = x - left

    # Directional zones
    up_zone = local_view[:center_y, :, :]
    down_zone = local_view[center_y+1:, :, :]
    left_zone = local_view[:, :center_x, :]
    right_zone = local_view[:, center_x+1:, :]

    # to process ghosts and pellets
    zones = [up_zone, right_zone, left_zone, down_zone]
    features = []

    for zone in zones:
        ghosts = count_all_ghosts(zone)
        if ghosts > 0:
            features.append(0)
        else:
            pellets = count_pellets(zone)
            if pellets > 0:
                features.append(2)
            else:
                features.append(1)
    #show_local_view(local_view)
    #print(features)
    return tuple(features)

######################################################## EPSILON_GREEDY ########################################################

def epsilon_greedy(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(q_table[state])]

######################################################## MAIN PROGRAM ########################################################

learning_rate = 0.1
discount_factor = 0.95

epsilon = 1.0
epsilon_decay = 0.992
epsilon_min = 0.01

episodes = 5000
actions = [1, 2, 3, 4]  
reward_list = []
penalty_list = []

q_table = load_q_table("q_table.csv")

pacman_found_list = []

plt.ion()

for episode in range(episodes):

    print(f"Episode: {episode+1}")

    render = (episode + 1) % 1000 == 0
    #render = True
    env = make_env()
    obs, _ = env.reset()
    state = extract_features(obs)
    done = False
    randomA = False
    total_reward = 0
    last_reward = 0

    while not done:

        if randomA == False: 
            stuck = 0
            action = epsilon_greedy(state)
        else:
            stuck = -3
            action = random.choice(actions)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = extract_features(next_obs)
        
        # in case of being stuck, take a random action
        if state == next_state: randomA = True
        else: randomA = False
    
        total_reward += reward

        # penalties
        reward_penalty = -1 if reward == 0 else 2
        lives_penalty = -5 if info['lives'] == 0 else 0
        ghost_penalty = -10 if state[action - 1] == 0 else 1

        reward_q_table = reward + ghost_penalty + lives_penalty + reward_penalty + stuck

        # update q-table
        best_next = float(np.max(q_table[next_state]))
        q_table[state][action - 1] += learning_rate * (reward_q_table + discount_factor * best_next - q_table[state][action - 1])

        state = next_state
        obs = next_obs  

        last_reward = total_reward

        penalty_list.append(reward_q_table)

        if render:
            plt.imshow(obs)
            plt.title(f"Episodio {episode+1}")
            plt.axis("off")
            plt.pause(0.001)
            plt.clf()

    env.close()
    reward_list.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(reward_list[-10:])
        avg_penalty = np.mean(penalty_list[-10:])
        print(f"→ Episodio {episode+1} | Promedio: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
        print("Penalty mean: ", avg_penalty )


plt.ioff()

plt.figure(figsize=(10, 5))
plt.plot(reward_list, label="Recompensa por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("Evolución de la recompensa durante el entrenamiento")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("histogram_q_learning_agent.png")
plt.show()

# save q table 
with open('q_table.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['state', 'action_1', 'action_2', 'action_3', 'action_4'])  
    
    for state, action_values in q_table.items():
        writer.writerow([state] + list(action_values))

