import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from GreyScaleObs import GreyScaleObs
from CustomReward import CustomReward
from ModifyDim import ModifyDim
import matplotlib.pyplot as plot
from gymnasium.wrappers import ResizeObservation
from datetime import datetime
import numpy
import gc
import torch
import os
from CustomRewardMetrics import CustomRewardMetrics

def applyCustomWrappers(option = "train"):
    def _init():
        env = gym.make("ALE/Pacman-v5")
        if option == "train":
            env = CustomReward(env)
        else:
            env = CustomRewardMetrics(env)
        env = GreyScaleObs(env)
        env = ResizeObservation(env, (84, 84))
        env = ModifyDim(env)
        return env
    return _init

def applyWrappers(option="train"):
    if option == "train":
        env = SubprocVecEnv([applyCustomWrappers(option) for _ in range(4)])
    else:
        env = DummyVecEnv([applyCustomWrappers(option)])
    env = VecFrameStack(env, n_stack=4)
        
    return env

if __name__ == "__main__":
    #Train PPO
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    env = applyWrappers()
    evalCallbackEnv = applyWrappers(option="test")
    evalCallBack = EvalCallback(evalCallbackEnv, eval_freq=10000, deterministic=True, n_eval_episodes=15, best_model_save_path="./resultsppo/G3/", log_path="./logs/")
    model = PPO("CnnPolicy", env=env, n_steps=1536, batch_size=384, n_epochs=8, gamma=0.99, gae_lambda=0.9, clip_range=0.04, ent_coef=0.003, vf_coef=0.5, learning_rate=2.5e-4, max_grad_norm=0.5, tensorboard_log="./tensorboardPPOPacman_G3/", verbose=1, device="cuda")
    model.learn(total_timesteps=2000000, callback=evalCallBack)
    model.save("./resultsppo/G3/pacmanPPO_G3_" + date)
    env.close()
    del model 
    del env
    del evalCallbackEnv
    del evalCallBack
    gc.collect()
    torch.cuda.empty_cache()

    
    #Pruebas y extracción de métricas (NO lo renderizo)
    """
    env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode=None), option="test")
    
    for nombre in os.listdir("./resultsppo/T"):
        ruta = os.path.join("./resultsppo/T", nombre)
        print(nombre)
        iter = 101
        model = PPO.load("/home/tomas/Desktop/PacmanRL/DQN/resultsppo/T/" + nombre, env=env)
        rewardsVector = []
        obs, _ = env.reset(seed=2025)
        #print(nombre)
        for i in range(1, iter):
            rewardsSum = 0
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done:
                steps += 1
                action, _ = model.predict(numpy.array(obs), deterministic=True)
                #print(action)
                obs, reward, terminated, truncated, info = env.step(action)
                rewardsSum += reward
                #if (reward != 0 and reward != 1 and reward != 2):
                #    print(reward)
                #    input()
                done = terminated or truncated
            if info["lives"] > 0:
                print("Ganador con ", steps, " pasos y una recompensa de ", rewardsSum )
            rewardsVector.append(rewardsSum)

        sum = 0
        for i in rewardsVector:
            sum += i
        sum = sum / len(rewardsVector)
        print(sum)
        print(max(rewardsVector))
        x = list(range(1, iter))
        plot.bar(x, rewardsVector)
        plot.xlabel("Episodio")
        plot.ylabel("Reward alcanzada")
        plot.title("Reward alcanzado por episodio")
        os.makedirs("graficos", exist_ok=True)
        os.makedirs("graficos/testsT", exist_ok=True)
        plot.savefig("graficos/testsT/rewards" + nombre + ".png")
        plot.close()
        env.close()
    """