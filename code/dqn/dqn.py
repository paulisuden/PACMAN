import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, ActionWrapper, spaces
from gymnasium.spaces import Box
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
import matplotlib.pyplot as plt

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

######################################################## LIMIT ACTIONS ########################################################

class LimitActionsWrapper(ActionWrapper):
    """
    Wrapper que transforma un espacio de acciones Discrete(8) del agente
    en acciones válidas del entorno [1, 8].
    """
    def __init__(self, env):
        super().__init__(env)
        self.valid_actions = [1, 2, 3, 4, 5, 6, 7, 8]  # up, right, left, down, upright, upleft, downrigth, downleft
        self.action_space = spaces.Discrete(len(self.valid_actions))

    def action(self, action):
        # Convierte acción del agente (0-7) a (1-8)
        return self.valid_actions[action]

######################################################## MAKE ENVIRONMENT ########################################################

# --- Make env compatible with SB3 ---
def make_env():
    def _init():
        env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array") 
        env = LimitActionsWrapper(env)      
        env = GrayScaleObservation(env, keep_dim=True)  
        env = CropObservation(env, bottom=38)
        env = ResizeObservation(env, shape=(84, 84))  
        return env
    return _init

######################################################## REWARD ########################################################

""" guarda las recompensas de cada episodio """
class RewardTrackerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

##################################################################################################################################

# Vectorizar entorno (para que SB3 funcione)
vec_env = DummyVecEnv([make_env()])

# crear y entrenar modelo DQN 
model = DQN(
    # red convolucional
    policy="CnnPolicy",
    env=vec_env,
    learning_rate=1e-4,
    # cuántas transiciones guarda el replay buffer para entrenar
    buffer_size=50000,
    # empieza a entrenar a partir de 1000
    learning_starts=1000,
    # cuántas transiciones usa por paso de entrenamiento
    batch_size=32,
    gamma=0.99,
    # cada cuántos pasos de entorno se hace 1 paso de entrenamiento
    train_freq=4,
    # cada cuántos pasos se actualiza la target network
    target_update_interval=1000,
    verbose=1,
    # donde guarda los datos de TensorBoard.
    tensorboard_log="./dqn_pacman_tensorboard/"
)

reward_callback = RewardTrackerCallback()

# --- Entrenar ---
model.learn(total_timesteps=100_000, callback=reward_callback) 

# --- Guardar modelo ---
model.save("dqn_pacman")

# --- Evaluar modelo entrenado ---
obs, _ = vec_env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = vec_env.step(action)
    done = terminated or truncated

    """
    # Mostrar imagen
    frame = obs[0].squeeze()  # (80, 80)
    plt.imshow(frame, cmap='gray')  
    plt.axis("off")
    plt.pause(0.05)  # velocidad de reproducción
    """


#Graficar recompensas
plt.plot(reward_callback.episode_rewards)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("Recompensa por episodio (DQN Pacman)")
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_pacman_rewards.png")
plt.show()

