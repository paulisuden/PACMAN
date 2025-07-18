import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from GreyScaleObs import GreyScaleObs
from CustomReward import CustomReward
import matplotlib.pyplot as plot
from gymnasium.wrappers import ResizeObservation, FrameStack
from datetime import datetime
import numpy
import gc
import torch
import csv
import os
from CustomRewardDefinitiveMetrics import CustomRewardDefinitiveMetrics

#print(torch.version.hip) 
#print(torch.cuda.is_available()) 
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def applyWrappers(env, option="train"):
    if option == "train":
        env = Monitor(env)
        env = CustomReward(env)
    else:
        env = Monitor(env)
        env = CustomRewardDefinitiveMetrics(env)
    env = GreyScaleObs(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, num_stack=4)
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
"""
env = gym.make("ALE/Pacman-v5", frameskip=4, render_mode="human")
while True:
    
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
"""

#Train con DQN 
"""
env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode="rgb_array"))
evalCallbackEnv = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode="rgb_array"), option="test")

evalCallBack = EvalCallback(evalCallbackEnv, eval_freq=100000, deterministic=True, n_eval_episodes=15, best_model_save_path="./evalResults/4/", log_path="./logs/")
model = DQN("CnnPolicy", env = env, buffer_size= 200000, batch_size = 32, verbose=1, learning_rate=5e-5, exploration_fraction=0.15, exploration_final_eps=0.05, tensorboard_log="./tensorboardDQNPacman/" + date, device="cuda")

model.learn(total_timesteps=12000000, callback=evalCallBack)
model.save("pacmanDqn" + date)

env.close()
del model
del env
evalCallbackEnv.close()
del evalCallbackEnv
del evalCallBack
gc.collect()
torch.cuda.empty_cache()

"""
#Test con renderización
"""
env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode="human"), option="test")
model = DQN.load("./evalResults/pacmanDqn2025-07-09_03-13-38", env=env)
print("El directorio actual es ", os.getcwd())
lista = []
for i in range(0, 1):
    #Test
    obs, _ = env.reset(seed=2025)
    done = False
    sum = 0
    while not done:
        action, _ = model.predict(numpy.array(obs), deterministic=True)
        #print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        sum = sum + reward
        #if (reward != 0 and reward != 1 and reward != 2):
        #    print(reward)
        #    input()
        done = terminated or truncated
    if info["lives"] > 0:
        lista.append(i)
        print(lista)
env.close()
print(lista)
"""

#Para extraer métricas, NO se renderiza
"""
env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, render_mode=None), option="test")

for nombre in os.listdir("./results8"):
    ruta = os.path.join("./results8", nombre)
    print(nombre)
    iter = 101
    model = DQN.load("/home/tomas/Desktop/PacmanRL/DQN/results8/" + nombre, env=env)
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
            print("Ganó usando ", steps, " pasos y una recompensa de ", rewardsSum )
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
    os.makedirs("graficos/tests15", exist_ok=True)
    plot.savefig("graficos/tests15/rewards" + nombre + ".png")
    plot.close()
    env.close()
"""


env = applyWrappers(gym.make("ALE/Pacman-v5", frameskip=4, mode=5), option="test")

for nombre in os.listdir("./tests/bestResultsDQN"):
    ruta = os.path.join("./tests/bestResultsDQN", nombre)
    print(nombre)
    iter = 101
    model = DQN.load(ruta, env=env)
    rewardsVector = []
    ghostsList = []
    pointsList = []
    bigPointsList = []
    stepsList = []
    wins = []
    winrate = 0
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
        if points >= 126:
            wins.append(1)
            winrate += 1
        else:
            wins.append(0)
        

    print(max(rewardsVector))
    nombre = nombre[:-4]
    plotStats("Métrica integradora", "Métrica integradora alcanzada por episodio en modo 5", iter, rewardsVector, "rewardsDQN" + nombre)
    plotStats("Ghosts", "Fantasmas comidos por episodio en modo 5", iter, ghostsList, "ghostsDQN" + nombre)
    plotStats("Points", "Puntos comidos por episodio en modo 5", iter, pointsList, "pointsDQN" + nombre)
    plotStats("Big points", "Puntos grandes comidos por episodio en modo 5", iter, bigPointsList, "bigPointsDQN" + nombre)
    plotStats("Steps", "Pasos dados por episodio en modo 5", iter, stepsList, "stepsDQN" + nombre)
    plotBoxAndWhiskers(rewardsVector, "Reward", "rewardsDQN" + nombre, iter, "Boxplot de reward por episodio en modo 5")
    plotBoxAndWhiskers(ghostsList, "Ghosts", "ghostsDQN" + nombre, iter, "Boxplot de fantasmas comidos por episodio en modo 5")
    plotBoxAndWhiskers(pointsList, "Points", "pointsDQN" + nombre, iter, "Boxplot de puntos comidos por episodio en modo 5")
    plotBoxAndWhiskers(bigPointsList, "Big Points", "bigPointsDQN" + nombre, iter, "Boxplot de puntos grandes comidos por episodio en modo 5")
    plotBoxAndWhiskers(stepsList, "Steps", "stepsDQN" + nombre, iter, "Boxplot de pasos dados por episodio en modo 5")

    os.makedirs("graficos", exist_ok=True)
    with open("graficos/meansDQN" + nombre + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Métrica", "Promedio"])
        writer.writerow(["Métrica integradora", sum(rewardsVector) / len(rewardsVector)])
        writer.writerow(["Ghosts", sum(ghostsList) / len(ghostsList)])
        writer.writerow(["Points", sum(pointsList) / len(pointsList)])
        writer.writerow(["Big Points", sum(bigPointsList) / len(bigPointsList)])
        writer.writerow(["Steps", sum(stepsList) / len(stepsList)])
        writer.writerow(["Victorias", winrate / len(stepsList)])

    with open("graficos/resultsDQN" + nombre + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episodio", "Métrica integradora", "Ghosts", "Points", "Big Points", "Steps", "Ganó"])
        for i in range(len(rewardsVector)):
            writer.writerow([i + 1, rewardsVector[i], ghostsList[i], pointsList[i], bigPointsList[i], stepsList[i], wins[i]])
    env.close()


