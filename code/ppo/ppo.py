import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from GreyScaleObs import GreyScaleObs
from CustomReward import CustomReward
import csv
from ModifyDim import ModifyDim
import matplotlib.pyplot as plot
from gymnasium.wrappers import ResizeObservation, FrameStack
from datetime import datetime
import numpy
import gc
import torch
import os
from CustomRewardDefinitiveMetrics import CustomRewardDefinitiveMetrics

def applyCustomWrappers(option = "train"):
    def _init():
        env = gym.make("ALE/Pacman-v5")
        if option == "train":
            env = CustomReward(env)
        else:
            env.reset(seed=2025)
            env = CustomRewardDefinitiveMetrics(env)
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

def applyWrappersMetrics(env, option="train"):
    if option == "train":
        env = Monitor(env)
        env = CustomReward(env)
    else:
        env = Monitor(env)
        env = CustomRewardDefinitiveMetrics(env)
    env = GreyScaleObs(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, num_stack=4)
    #env = ModifyDim(env)
    return env

def plotStats(ylabel, title, iter, vector, filename):
    x = list(range(1, iter))
    plot.bar(x, vector)
    plot.xlabel("Episodio")
    plot.ylabel(ylabel)
    plot.title(title)
    os.makedirs("graficos", exist_ok=True)
    plot.savefig("graficos/" + filename + ".png")
    plot.close()

def plotBoxAndWhiskers(vector, ylabel, filename, iter, title):
    x = list(range(1, iter))
    plot.boxplot(vector, vert=True)
    plot.ylabel(ylabel)
    plot.title(title)
    os.makedirs("graficos", exist_ok=True)
    plot.savefig("graficos/boxplot_" + filename + ".png")
    plot.close()

if __name__ == "__main__":
    """
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

    """
    #Pruebas y extracción de métricas (NO lo renderizo)
    """
    env = applyWrappersMetrics(gym.make("ALE/Pacman-v5", frameskip=4, render_mode=None), option="test")
    
    for nombre in os.listdir("./resultsppo/1"):
        ruta = os.path.join("./resultsppo/1", nombre)
        print(nombre)
        iter = 101
        model = PPO.load("/home/tomas/Desktop/PacmanRL/PPO/resultsppo/1/" + nombre, env=env)
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
                print("Winner with ", steps, " steps and reward of ", rewardsSum )
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
        os.makedirs("graficos/testsdefinitivos", exist_ok=True)
        plot.savefig("graficos/testsdefinitivos/rewards" + nombre + ".png")
        plot.close()
        env.close()
    """

    env = applyWrappersMetrics(gym.make("ALE/Pacman-v5", frameskip=4, render_mode=None), option="test")

    for nombre in os.listdir("./tests/bestResultsPPO"):
        ruta = os.path.join("./tests/bestResultsPPO", nombre)
        print(nombre)
        iter = 101
        model = PPO.load(ruta, env=env)
        rewardsVector = []
        ghostsList = []
        pointsList = []
        bigPointsList = []
        stepsList = []
        obs, _ = env.reset(seed=2025)
        #print(nombre)
        for i in range(1, iter):
            rewardsSum = 0
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(numpy.array(obs), deterministic=True)
                #print(action)
                obs, reward, terminated, truncated, info = env.step(action)
                #if (reward != 0 and reward != 1 and reward != 2):
                #    print(reward)
                #    input()
                done = terminated or truncated
            totalSteps = info.get("totalSteps")
            points = info.get("points")
            bigPoint = info.get("bigPoint")
            ghosts = info.get("ghosts")
            rewardsSum = ghosts * 5 + points + bigPoint * 3
            if info["lives"] > 0:
                print("Ganó usando ", totalSteps, " pasos y una recompensa de ", rewardsSum )
            rewardsVector.append(rewardsSum)
            ghostsList.append(ghosts)
            pointsList.append(points)
            bigPointsList.append(bigPoint)
            stepsList.append(totalSteps)

        print(max(rewardsVector))

        plotStats("Reward", "Reward alcanzada por episodio", iter, rewardsVector, "rewardsPPO" + nombre)
        plotStats("Ghosts", "Fantasmas comidos por episodio", iter, ghostsList, "ghostsPPO" + nombre)
        plotStats("Points", "Puntos comidos por episodio", iter, pointsList, "pointsPPO" + nombre)
        plotStats("Big points", "Puntos grandes comidos por episodio", iter, bigPointsList, "bigPointsPPO" + nombre)
        plotStats("Steps", "Pasos dados por episodio", iter, stepsList, "stepsPPO" + nombre)
        plotBoxAndWhiskers(rewardsVector, "Reward", "rewardsPPO" + nombre, iter, "Boxplot de reward por episodio")
        plotBoxAndWhiskers(ghostsList, "Ghosts", "ghostsPPO" + nombre, iter, "Boxplot de fantasmas comidos por episodio")
        plotBoxAndWhiskers(pointsList, "Points", "pointsPPO" + nombre, iter, "Boxplot de puntos comidos por episodio")
        plotBoxAndWhiskers(bigPointsList, "Big Points", "bigPointsPPO" + nombre, iter, "Boxplot de puntos grandes comidos por episodio")
        plotBoxAndWhiskers(stepsList, "Steps", "stepsPPO" + nombre, iter, "Boxplot de pasos dados por episodio")

        os.makedirs("graficos", exist_ok=True)
        with open("graficos/meansPPO2" + nombre + ".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Métrica", "Promedio"])
            writer.writerow(["Reward", sum(rewardsVector) / len(rewardsVector)])
            writer.writerow(["Ghosts", sum(ghostsList) / len(ghostsList)])
            writer.writerow(["Points", sum(pointsList) / len(pointsList)])
            writer.writerow(["Big Points", sum(bigPointsList) / len(bigPointsList)])
            writer.writerow(["Steps", sum(stepsList) / len(stepsList)])

        with open("graficos/resultsPPO2" + nombre + ".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episodio", "Reward", "Ghosts", "Points", "Big Points", "Steps"])
            for i in range(len(rewardsVector)):
                writer.writerow([i + 1, rewardsVector[i], ghostsList[i], pointsList[i], bigPointsList[i], stepsList[i]])
        env.close()
