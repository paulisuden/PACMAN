import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def make_env(apply_wrappers=True):
    env = gym.make("ALE/Pacman-v5")
    if apply_wrappers:
        #env = GrayScaleObservation(env)                 # grey scale
        env = CropObservation(env, bottom=38)           # crop 
        env = ResizeObservation(env, shape=(84,84))   # resize 
    return env


#####################

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
            tolerance=10,
            excluded_colors=EXCLUDED_COLORS, 
            exclude_tol=15
        )
    return total

# imprimir RGB al hacer clic
def click_event(event, x, y, a, b):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convertir coordenadas al tamaño original
        orig_x = x // scale
        orig_y = y // scale
        color = obs[orig_y, orig_x]
        print(f"Coordenadas: ({orig_x}, {orig_y}) - Color RGB: {color}")
    
# Mostrar local_view con click interactivo
def show_local_view_with_click(obs, scale=40):
    h, w, _ = obs.shape
    PACMAN_MAIN_COLOR = np.array([252, 224, 144])
    mask_pacman = np.all(np.abs(obs - PACMAN_MAIN_COLOR) <= 10, axis=-1)
    pos = np.argwhere(mask_pacman)

    if len(pos) == 0:
        y, x = h // 2, w // 2
    else:
        y, x = pos[0]

    # centered crop
    window_size = 19
    half = window_size // 2
    top = max(0, y - half)
    bottom = min(h, y + half + 1)
    left = max(0, x - half)
    right = min(w, x + half + 1)
    local_view = obs[top:bottom, left:right, :]

    # Redimensionar para visualización con OpenCV
    local_view_big = cv2.resize(local_view, (local_view.shape[1]*scale, local_view.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

    # Evento de clic
    def click_event_local(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = x // scale
            orig_y = y // scale
            if 0 <= orig_y < local_view.shape[0] and 0 <= orig_x < local_view.shape[1]:
                color = local_view[orig_y, orig_x]
                print(f"Color RGB: {color}")

    cv2.namedWindow("Vista Local")
    cv2.setMouseCallback("Vista Local", click_event_local)
    cv2.imshow("Vista Local", cv2.cvtColor(local_view_big, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#####################

# Crear entorno
env = make_env()
obs, _ = env.reset()
# Usar esta función
show_local_view_with_click(obs)
