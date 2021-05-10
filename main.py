import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrappers import wrapper

# stable baselines
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.evaluation import evaluate_policy

import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)

models = [DQN.load("models/dqn")]

model_names = ["deep q-learning"]

for i in range(len(models)):
    cr = 0

    print("Learning to beat super mario bros using {}".format(model_names[i]))

    obs = env.reset() 

    while True:
        action, _states = models[i].predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        cr += rewards
        print("Reward: {}\t\t".format(cr), end='\r')
        env.render()
        if (done):
            print("Finished an episode with total reward: ", cr)
            cr = 0
            break

    print(evaluate_policy(model, env, n_eval_episodes=10, deterministic=False, render=True))

    print("Done.")