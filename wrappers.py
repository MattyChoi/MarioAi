# basic wrappers, useful for reinforcement learning on gym envs

from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv

def wrapper(env):
    # Wrap environment using wrappers used for atari games
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = MaxAndSkipEnv(env, skip=8)
    return env
