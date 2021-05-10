from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnCnnPolicy
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks import ProgressBarManager
import tensorflow as tf
import cv2
import os
import argparse





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
    env = ClipRewardEnv(env)
    return env




# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def run(run_name, existing_model):

    # Create log dir
    log_dir = "./dqn_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("\n-----------------------Setting up environment-----------------------")
    
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = wrapper(env)

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=25000, save_path='../models/',
                                            name_prefix=run_name)

    eval_callback = EvalCallback(env,
                                best_model_save_path='../models/',
                                log_path='../models/',
                                eval_freq=250000,
                                deterministic=True,
                                render=False)

    print("\n-----------------------Compiling model-----------------------")

    if existing_model:
        try:
            model = DQN.load(existing_model, env, tensorboard_log="../mario_tensorboard/")
        except:
            print(f"{existing_model} does not exist!")
            exit(0)
    else:
        model = DQN(LnCnnPolicy,
                    env,
                    batch_size=128, # Optimizable (higher batch sizes ok according to https://arxiv.org/pdf/1803.02811.pdf)
                    verbose=1, 
                    learning_starts=10000,
                    learning_rate=1e-4,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.1,
                    prioritized_replay=True, 
                    prioritized_replay_alpha=0.6,
                    train_freq=8,
                    target_network_update_freq=100000,
                    tensorboard_log="../mario_tensorboard/"
                )

    print("\n-----------------------Start Training-----------------------")

    time_steps = 500000

    with ProgressBarManager(time_steps) as progress_callback:
        model.learn(total_timesteps=time_steps,
                    log_interval=1,
                    callback=[progress_callback, checkpoint_callback, eval_callback],
                    tb_log_name=run_name)

    print("\n---------------------Done! Saving Model---------------------")
    model.save("models/dqn".format(run_name))


def test_env(env, frame_by_frame=False):
    obs = env.reset()
    while True:
        obs, rewards, dones, info = env.step(env.action_space.sample())
        print(obs._frames)
        if (frame_by_frame):
            cv2.imshow("frames", obs._frames[0])
            cv2.waitKey()
        else:
            env.render()
        print("reward:", rewards)
        print("timestep:", info['timestep'])










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-existing", nargs='?', help="Train existing model")
    args = parser.parse_args()

    run("dqn", args.train_existing)