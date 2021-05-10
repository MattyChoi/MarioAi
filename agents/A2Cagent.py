from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import A2C
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
# from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks import ProgressBarManager
import tensorflow as tf
import cv2
import os
import argparse

# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def run(run_name, existing_model):

    # Create log dir
    log_dir = "./monitor_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("\n-----------------------Setting up environment-----------------------")
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = EpisodicLifeEnv(env)

    # Preprocessing
    env = WarpFrame(env)
    env = FrameStack(env, n_frames = 4)

    # Evaluate every kth frame and repeat action
    env = MaxAndSkipEnv(env, skip = 8)

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=25000, save_path='./models/',
                                            name_prefix=run_name)

    eval_callback = EvalCallback(env,
                                best_model_save_path='./models/',
                                log_path='./models/',
                                eval_freq=250000,
                                deterministic=True,
                                render=False)

    print("\n-----------------------Compiling model-----------------------")

    if existing_model:
        try:
            model = A2C.load(existing_model, env, tensorboard_log="./mario_tensorboard/")
        except:
            print(f"{existing_model} does not exist!")
            exit(0)
    else:
        model = A2C(CnnPolicy,
                    env,
                    verbose=1, 
                    learning_rate=1e-4,
                    tensorboard_log="./mario_tensorboard/"
                )

    print("\n-----------------------Start Training-----------------------")

    time_steps = 100000

    with ProgressBarManager(time_steps) as progress_callback:
        model.learn(total_timesteps=time_steps,
                    log_interval=1,
                    callback=[progress_callback, checkpoint_callback, eval_callback],
                    tb_log_name=run_name)

    print("\n---------------------Done! Saving Model---------------------")
    model.save("models/a2c".format(run_name))
    env.render()







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

    run("a2c", args.train_existing)