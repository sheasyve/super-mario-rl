import argparse
import os
import random
from time import time

import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

CHECKPOINT_DIR, LOG_DIR = "./train/", "./logs/"
# Set render_mode to 'human' to display the model on screen as it learns
# Set render_mode to 'rgb_array' for large increase in training speed (40% in my case)
# render_mode = 'human'
render_mode = "rgb_array"
# default will change if -mp flag called
processes = 1


class TrainAndLoggingCallback(BaseCallback):
    # Setup model saving callback
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "best_model_{}".format(self.n_calls)
            )
            self.model.save(model_path)
        return True


def view_initial_images(env):
    # Draw the initial preprocessed screen (first 4 stacked frames)
    state = env.reset()
    state, reward, done, info = env.step([5])
    plt.figure(figsize=(20, 16))
    for idx in range(state.shape[3]):
        plt.subplot(1, 4, idx + 1)
        plt.imshow(state[0][:, :, idx])
    plt.show()


def make_env(stages: list, random: bool, rank: int, seed: int = 0):
    def _init():
        # env =  gym_super_mario_bros.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode='human')
        if random:
            env = gym_super_mario_bros.make(
                "SuperMarioBrosRandomStages-v0",
                apply_api_compatibility=True,
                render_mode=render_mode,
            )
        else:
            env = gym_super_mario_bros.make(
                "SuperMarioBrosRandomStages-v0",
                apply_api_compatibility=True,
                render_mode=render_mode,
                stages=stages,
            )
        # env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0", apply_api_compatibility=True, render_mode='human')
        env.reset(seed=seed + rank)
        # env = ResizeObservation(env, 64)
        env = GrayScaleObservation(env, keep_dim=True)
        return env

    set_random_seed(seed)
    return _init


def env_setup(stages, random=False, multiproc=False):
    if multiproc:
        env = SubprocVecEnv(
            [make_env(rank=i, stages=stages, random=random) for i in range(processes)]
        )
    else:
        if random:
            env = gym_super_mario_bros.make(
                "SuperMarioBrosRandomStages-v0",
                apply_api_compatibility=True,
                render_mode=render_mode,
            )
        else:
            env = gym_super_mario_bros.make(
                "SuperMarioBrosRandomStages-v0",
                apply_api_compatibility=True,
                render_mode=render_mode,
                stages=stages,
            )
            # env = MonitorWrapper(env)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        # env = ResizeObservation(env, 64)
        env = GrayScaleObservation(env, keep_dim=True)
        env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")
    env = VecNormalize(
        env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-08,
    )
    return env


def policy(
    env, callback, filename="archive/temp.model", learn=False, learning_rate=0.0001
):
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=learning_rate,  # EXPERIMENT:Adjust this learning rate. So far we have found .0001 to be best.
        n_steps=512,
    )
    if learn:
        model.learn(
            total_timesteps=50000, callback=callback
        )  # Train the AI model. EXPERIMENT:train for longer periods of time. We have found it is best to use at least 50k.
    model.save(filename)
    return model


def train(
    model_name,
    env,
    callback,
    iterations_per_training_interval=3,
    training_iterations=5,
    test_steps=512 * 3,
    max_training_time=None,  # Int in seconds.
    end_condition="iterations",
):
    # Trains a given model for iterations_per_training_interval itrerations, then tests the model and records the average reward.
    # Repeats this process training_iterations times.
    rewards = []

    i = 0
    if end_condition == "iterations":
        i_bound = len(training_iterations)
    elif end_condition == "time":
        i_bound = max_training_time
        start_time = time()
    else:
        raise Exception("""end_condition should be "iterations" or "time".""")
    while i < i_bound:
        # Load the model
        model = PPO.load("./" + model_name, env=env)
        model.learn(
            total_timesteps=512 * iterations_per_training_interval,
            callback=callback,
            progress_bar=True,
        )
        model.save(model_name)

        # Test the model and record the average reward
        total_rewards, i = 0, 0
        state = env.reset()
        while i < test_steps:
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)
            total_rewards += reward
            i += 1
        avg_reward = (total_rewards / test_steps)[0]
        rewards.append(avg_reward)
        print(f"avg reward: {avg_reward}")
        if end_condition == "iterations":
            i += 1
        elif end_condition == "time":
            i = time() - start_time
    return rewards


def plot(rewards, iterations_per_training_interval):
    training_iterations = len(rewards)
    x_ticks = list(range(0, training_iterations))
    fig, ax = plt.subplots()
    ax.plot(x_ticks, rewards)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.title(
        "Average reward over"
        f" {training_iterations  * iterations_per_training_interval} iterations"
    )
    plt.xlabel("Iteration number", color="black", size=28)
    plt.ylabel("Average reward", color="black", size=28)
    plt.plot(x_ticks, rewards, marker="o")
    plt.savefig("./archive/plots/final.png")
    plt.show()


def final_test(fn, env):
    # Renders the finished model playing a game of mario
    model = PPO.load("./" + fn, env=env)
    done = True
    state = env.reset()
    for step in range(100000):
        action, _ = model.predict(state)
        if done:
            env.reset()
        observation, reward, done, info = env.step(action)
        env.render()
    env.close()


def parsing():
    parser = argparse.ArgumentParser(description="Train RL models for Super Mario")
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        metavar="",
        default="archive/init_model",
        help="Filename to begin training model with. Recommended for continious mode.",
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help=(
            "Train the model with stage 1-1 and 4 random level combinations. From world"
            " 1-8 and stage 1-4"
        ),
    )
    parser.add_argument(
        "-tr",
        "--truerandom",
        action="store_true",
        help="Train the model with a random stage at every death and reset",
    )
    parser.add_argument("-i", "--iterative", action="store_true", help="iterative mode")
    parser.add_argument(
        "-mp",
        "--multiproc",
        nargs="?",
        const=True,
        default=None,
        type=int,
        help=(
            "Enables multiprocesing based on detected number of cpu cores by default,"
            " pass a number to specify number of cpu cores manually (must be <= total"
            " detected cores)"
        ),
    )
    args = parser.parse_args()
    return args


def stageing(args):
    stages = ["1-1"]
    if args.random:
        for i in range(4):
            world = random.randint(1, 8)
            stage = random.randint(1, 4)
            stage_string = f"{world}-{stage}"
            stages.append(stage_string)
    return stages


def main():
    global processes
    args = parsing()
    stages = stageing(args)
    total_cpu_cores = os.cpu_count()
    processes = int(total_cpu_cores / 2)
    multiproc = False

    if args.multiproc == True:
        multiproc = True
    if args.multiproc != None and args.multiproc != True:
        processes = args.multiproc if processes <= total_cpu_cores else total_cpu_cores

    # Set up environment
    env = env_setup(stages=stages, random=args.truerandom, multiproc=multiproc)
    import pdb

    pdb.set_trace()
    # view_initial_images(env)
    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    # Test parameters
    fn = args.filename
    training_iterations, iterations_per_training_interval = 5, 50
    learning_rate = 0.0001

    # Create (and save) the initial model. Change "learn" to "True" if you want to skip the plotting step and train it normally
    model = policy(env, callback, filename=fn, learn=False, learning_rate=learning_rate)

    # Train, and obtain average rewards as iterations increase. Then plot these rewards
    rewards = train(
        fn,
        env,
        callback,
        training_iterations=None,
        iterations_per_training_interval=iterations_per_training_interval,
        end_condition="time",
        max_training_time=3600,
    )
    model.save("archive/final.model")
    plot(rewards, iterations_per_training_interval)
    # Perform an extensive test after model has been fully trained
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)#Evaluate model to get training rewards
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # final_test(fn,env)


if __name__ == "__main__":
    main()
"""
@misc{gym-super-mario-bros,
  author = {Christian Kauten},
  howpublished = {GitHub},
  title = {{S}uper {M}ario {B}ros for {O}pen{AI} {G}ym},
  URL = {https://github.com/Kautenja/gym-super-mario-bros},
  year = {2018},
}
"""
