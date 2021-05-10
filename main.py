import time
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import DQNAgent
from wrappers import wrapper

'''
# openai baselines
from baselines import deepq
from baselines import a2c
from baselines import acktr
from baselines import ppo2
'''
# stable baselines
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import MlpPolicy


# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)

# Parameters
states = (84, 84, 4)
actions = env.action_space.n

# Agent
agents = [DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)]

# Episodes
episodes = 500
rewards = []

# Timing
start = time.time()
step = 0

# Main loop
for agent in agents:
    for e in range(1, episodes + 1):

        # Reset env
        state = env.reset()

        # Reward
        total_reward = 0
        iter = 0

        # Play
        while True:

            # Run agent
            action = agent.run(state=state)

            # Perform action
            next_state, reward, done, info = env.step(action=action)

            # Remember
            agent.add(experience=(state, next_state, action, reward, done))

            # Replay
            agent.learn()

            # Total reward
            total_reward += reward

            # Update state
            state = next_state

            # Increment
            iter += 1

            # If done break loop
            if done or info['flag_get']:
                break

        # Rewards
        rewards.append(total_reward / iter)

        # Print
        if e % 1 == 0:
            print('Episode {e} - '
                'Frame {f} - '
                'Frames/sec {fs} - '
                'Epsilon {eps} - '
                'Mean Reward {r}'.format(e=e,
                                        f=agent.step,
                                        fs=np.round((agent.step - step) / (time.time() - start)),
                                        eps=np.round(agent.eps, 4),
                                        r=np.mean(rewards[-100:])))
            start = time.time()
            step = agent.step

    # Show env
    env.render()

    # Save rewards
    # np.save('rewards.npy', rewards)