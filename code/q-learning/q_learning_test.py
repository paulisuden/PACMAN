import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import ResizeObservation
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

def load_q_table(filename="q_table.csv"):
    q_table = defaultdict(lambda: np.zeros(4))
    if os.path.exists(filename):
        print("archivo encontrado")
        with open(filename, mode='r') as f:
            reader = csv.reader(f)
            next(reader)  # jump header
            for row in reader:
                state = eval(row[0])  # convert a string "(1, 0, 0, 1)" to a tuple
                values = list(map(float, row[1:]))
                q_table[state] = np.array(values)
    else:
        print("archivo no encontrado")
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

def make_env(render):
    if render:
        env = gym.make("ALE/Pacman-v5", render_mode="human")
    else:
        env = gym.make("ALE/Pacman-v5")
    env = CropObservation(env, bottom=38)           # crop 
    env = ResizeObservation(env, shape=(84,84))     # resize 
    return env

######################################################## EXTRACT FEATURES #########################################################

GHOST_COLOR = np.array([252, 144, 200])

PELLET_COLORS = [
    np.array([161, 141, 134]), 
    np.array([122, 109, 149]), 
    np.array([181, 157, 127]), 
    np.array([174, 152, 129])]

PACMAN_COLORS = [
    np.array([233, 208, 147]),
    np.array([157, 142, 159]),
    np.array([191, 171, 154]),
    np.array([252, 224, 144]),
    np.array([90, 84, 170]),
    np.array([206, 183, 139]),
    np.array([147, 134, 158]),
    np.array([239, 213, 146]),
    np.array([214, 191, 150]),
    np.array([189, 170, 154]),
    np.array([113, 104, 166]),
    np.array([169, 150, 148]),
    np.array([131, 118, 154])]

WALL_COLORS = [
    np.array([223, 192, 111]),
    np.array([145, 128, 140])
]

EXCLUDED_COLORS = PACMAN_COLORS + PELLET_COLORS

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
    total += count_color(zone, GHOST_COLOR, 20)
    return total

def is_the_wall(zone):
    total = 0
    for color in WALL_COLORS:
        total += count_color_excluding_multiple(
            zone,
            color_rgb=color,
            tolerance=25,
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
    window_size = 15
    half = window_size // 2
    top = max(0, y - half)
    bottom = min(h, y + half + 1)
    left = max(0, x - half)
    right = min(w, x + half + 1)
    local_view = obs[top:bottom, left:right, :]

    center_y = y - top
    center_x = x - left

    # Directional zones
    step = 5  # o 1 o 3, según resolución y tamaño
    up_zone    = local_view[center_y - step:center_y, center_x - 1:center_x + 2]
    down_zone  = local_view[center_y + 1:center_y + 1 + step, center_x - 1:center_x + 2]
    left_zone  = local_view[center_y - 1:center_y + 2, center_x - step:center_x]
    right_zone = local_view[center_y - 1:center_y + 2, center_x + 1:center_x + 1 + step]


    # to process ghosts and pellets
    zones = [up_zone, right_zone, left_zone, down_zone]
    features = []

    for zone in zones:
        ghosts = count_all_ghosts(zone)
        if ghosts > 0:
            features.append(0)
        else:
            wall = is_the_wall(zone)
            if wall > 0:
                features.append(1)
            else:
                features.append(2)
    #show_local_view(local_view)
    #print(features)
    return tuple(features)


######################################################## EPSILON_GREEDY ########################################################

def epsilon_greedy(state, num):
    if num == 0:
        return actions[np.argmax(q_table[state])]
    else:
        #devuelvo el segundo valor mayor
        q_values = q_table[state]
        sorted_indices = np.argsort(q_values)[::-1]
        return actions[sorted_indices[1]]

######################################################## MAIN PROGRAM ########################################################

episodes = 15
actions = [1, 2, 3, 4]  

reward_list = []
lives_list = []
results = []

q_table = load_q_table("q_table.csv")

pacman_found_list = []

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
    state = extract_features(obs)
    second_action = False
    while not done:

        if second_action: 
            action = epsilon_greedy(state, 1)
            stuck = 0
        else: action = epsilon_greedy(state, 0)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        new_state = extract_features(obs)

        if state == new_state and state != (1, 2, 2, 1): stuck += 1
        else: stuck = 0
        
        if stuck >= 2: second_action = True
        else: second_action = False

        total_reward += reward
        state = new_state

    lives = info["lives"]
    if lives > 0: win = 1 
    else: win = 0

    lives_list.append(lives)
    results.append({
        "episode": episode + 1,
        "score": total_reward,
        "lives_left": lives,
        "win": win
    })
    env.close()


df = pd.DataFrame(results)
df.to_csv("barplot_q_learning_results.csv", index=False)

plt.figure(figsize=(14, 4))
episodes_range = df["episode"]
plt.subplot(1, 3, 1)
plt.bar(episodes_range, df["score"], color="skyblue", edgecolor="black")
plt.title("Score")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.xticks(episodes_range)

plt.subplot(1, 3, 3)
plt.bar(episodes_range, df["lives_left"], color="green", edgecolor="black")
plt.title("Lives Left")
plt.xlabel("Episode")
plt.ylabel("Lives Left")
plt.xticks(episodes_range)

plt.tight_layout()
plt.savefig("barplot_q_learning.png")
plt.show()

