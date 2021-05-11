# basic wrappers, useful for reinforcement learning on gym envs

from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, ClipRewardEnv, FireResetEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY


def wrapper(env):
    """Apply a common set of wrappers for Atari games."""

    # Use actions only in the actions.RIGHT_ONLY array
    env = JoypadSpace(env, RIGHT_ONLY)

    # Evaluate every kth frame and repeat action
    env = MaxAndSkipEnv(env, skip=4)

    # preprocessing
    if 'FIRE' in env.unwrapped.get_action_meanings():
       env = FireResetEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    return env