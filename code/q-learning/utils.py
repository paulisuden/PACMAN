import gymnasium as gym
import numpy as np
from collections import defaultdict
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import ResizeObservation
import matplotlib.pyplot as plt
import csv
import os

"""Carga la q_table si existe y si no crea una nueva inicializada en 0"""
def load_q_table(filename="code/q-learning/q_table.csv"):
    q_table = defaultdict(lambda: np.zeros(4)) 
    if os.path.exists(filename):
        print(f"Archivo encontrado: {filename}")
        with open(filename, mode='r') as f:
            reader = csv.reader(f)
            next(reader)  
            for row in reader:
                try:
                    state = eval(row[0]) 
                    values = list(map(float, row[1:]))
                    q_table[state] = np.array(values)
                except Exception as e:
                    print(f"Error al leer fila {row}: {e}")
    else:
        print(f"Archivo no encontrado: {filename}")
    return q_table

######################################################## CROP OBSERVATION ########################################################

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

######################################################## EXTRACT FEATURES ########################################################

"""Para chequear el recorte que se hace en extract_features"""
def show_local_view(local_view):
    plt.imshow(local_view)
    plt.title("Local View centrada en Pac-Man")
    plt.axis('off') 
    plt.show()

"""Colores de los fantasmas, los puntos, el pacman y la pared"""

GHOST_COLOR = np.array([252, 144, 200])

PELLET_COLORS = [
    np.array([157, 138, 136]),
    np.array([184, 160, 126]),
    np.array([165, 144, 133]),
    np.array([177, 154, 128]),
    np.array([101, 92, 157]),
    np.array([135, 120, 144]),
    np.array([128, 114, 147]),
    np.array([115, 103, 152]),
    np.array([158, 138, 136]),
    np.array([104, 94, 156]),
    np.array([113, 101, 152]),
]

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

"""
Verifica si un bloque de píxeles está completamente rodeado por píxeles de color azul, para asegurarse de 
sea un pellet, y no la pared o el pacman
"""
def is_surrounded_by_blue(obs, x, y, h, w, blue_color, tolerance):
    for i in range(x - 1, x + h + 1):
        for j in range(y - 1, y + w + 1):
            if x <= i < x + h and y <= j < y + w:
                continue
            if i < 0 or j < 0 or i >= obs.shape[0] or j >= obs.shape[1]:
                return False
            if not np.all(np.abs(obs[i, j] - blue_color) <= tolerance):
                return False
    return True

"""
Verifica si todos los píxeles dentro de un bloque corresponden a algún color de pellet, 
dentro de una cierta tolerancia
"""
def block_matches_pellet_colors(block, pellet_colors, tolerance):
    h, w, _ = block.shape
    for i in range(h):
        for j in range(w):
            pixel = block[i, j]
            if not any(np.all(np.abs(pixel - pc) <= tolerance) for pc in pellet_colors):
                return False
    return True  

"""
Cuenta cuántos pellets hay en una zona determinada
Para cada posible forma de pellet recorre la zona y verifica:
    1. Si el bloque de píxeles coincide con los colores de los pellets
    2. Si ese bloque está rodeado por color azul
"""
def count_all_pellets(zone, pellet_colors, tolerance=25, blue_color=np.array([50, 50, 176]), blue_tol=45):
    count = 0
    pellet_shapes = [(3,1), (1,3), (1,2), (2,1), (2,2)]  

    h_img, w_img, _ = zone.shape

    for ph, pw in pellet_shapes:
        for i in range(h_img - ph + 1):
            for j in range(w_img - pw + 1):
                block = zone[i:i+ph, j:j+pw]
                if block_matches_pellet_colors(block, pellet_colors, tolerance):
                    if is_surrounded_by_blue(zone, i, j, ph, pw, blue_color, blue_tol):
                        count += 1
    return count

def count_color(zone, color_rgb, tolerance):
    return np.sum(np.all(np.abs(zone - color_rgb) <= tolerance, axis=-1))

def count_all_ghosts(zone):
    total = 0
    total += np.sum(np.all(np.abs(zone - GHOST_COLOR) <= 20, axis=-1))
    return total

"""Verifica que sea una pared excluyendo los colores de pacman y de los pellets"""
def count_color_excluding_multiple(zone, color_rgb, tolerance, excluded_colors, exclude_tol):
    # es true en cada píxel que está dentro de la tolerancia respecto al color_rgb
    match = np.all(np.abs(zone - color_rgb) <= tolerance, axis=-1)
    # quitamos las coincidencias con colores de pacman y la pared
    for excl in excluded_colors:
        excl_match = np.all(np.abs(zone - excl) <= exclude_tol, axis=-1)
        match = match & ~excl_match
    return np.sum(match)

"""Verifica si en esa zona hay una pared"""
def is_the_wall(zone):
    total = 0
    for color in WALL_COLORS:
        total += count_color_excluding_multiple(zone, color_rgb=color, tolerance=10,
            excluded_colors=EXCLUDED_COLORS, exclude_tol=15)
    return total

"""Extrae las caracteristicas de la observacion actual"""
def extract_features(obs, pacman_found_list):
    h, w, _ = obs.shape
    pacman_found = 0
    PACMAN_MAIN_COLOR = np.array([252, 224, 144])

    mask_pacman = np.all(np.abs(obs - PACMAN_MAIN_COLOR) <= 10, axis=-1)
    pos = np.argwhere(mask_pacman)
    
    if len(pos) == 0:
        y, x = h // 2, w // 2
    else:
        pacman_found = 1
        y, x = pos[0]

    pacman_found_list.append(pacman_found) 

    # Hace un recorte centrado en pcaman para reducir el tamaño de la observacion a analizar
    window_size = 15
    half = window_size // 2
    top = max(0, y - half)
    bottom = min(h, y + half + 1)
    left = max(0, x - half)
    right = min(w, x + half + 1)
    local_view = obs[top:bottom, left:right, :]

    center_y = y - top
    center_x = x - left

    # se definen las zonas a analizar excluyendo a pacman
    m = 2
    up_zone    = local_view[0:center_y - m, :]                     
    down_zone  = local_view[center_y + m + 1:, :]                     
    left_zone  = local_view[:, 0:center_x - m]                     
    right_zone = local_view[:, center_x + m + 1:]                     
    zones = [up_zone, right_zone, left_zone, down_zone]

    features = []

    for zone in zones:
        ghosts = count_all_ghosts(zone)
        if ghosts > 0:
            features.append(0)                      #* ghost
        else:
            pellets = count_all_pellets(zone, PELLET_COLORS)
            if pellets > 0:
                    features.append(3)              #* pellet
            else:
                wall = is_the_wall(zone)
                if wall > 0:
                    features.append(1)              #* pared
                else:
                    features.append(2)              #* libre
    return tuple(features), pacman_found_list