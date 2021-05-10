import numpy as np
import gym_super_mario_bros
from wrappers import wrapper

# stable baselines
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.evaluation import evaluate_policy

import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from train import *


# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = wrapper(env)


# create dqn, a2c, and ppo2 models
parser = argparse.ArgumentParser()
parser.add_argument("--train-existing", nargs='?', help="Train existing model")
args = parser.parse_args()
run("dqn", args.train_existing)
run("a2c", args.train_existing)
run("ppo2", args.train_existing)


models = [DQN.load("models/dqn"), A2C.load("models/a2c")]#, PPO2.load("models/ppo2")]

model_names = ["deep q-learning", "actor-critic"]#, "proximal policy optimization"]

for i in range(len(models)):
    cr = 0

    print("Learning to beat super mario bros using {}".format(model_names[i]))

    obs = env.reset() 
    env.render()
    env.render()
    env.render()
    env.render()
    env.render()
    env.render()
    env.render()

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