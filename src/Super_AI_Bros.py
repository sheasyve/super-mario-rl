# 1.SETUP MARIO
# Import the game
import gym_super_mario_bros

# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY 

# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace

#https://github.com/yumouwei/super-mario-bros-reinforcement-learning/blob/main/gym_utils.py
from gym_utils import SMBRamWrapper
from stable_baselines3.common.monitor import Monitor

import os

# # Import PPO for algos
from stable_baselines3 import PPO

# # Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

# 2.PREPROCESS ENVIRONMENT
# Setup game
# env = gym_super_mario_bros.make(
#     'SuperMarioBros-1-1-v0'
# )
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="rgb_array"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True


# 2.PREPROCESS ENVIRONMENT
# Install pytorch
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation

# # Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# # Import Vectorization Wrappers
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

# # 1. Create the base environment
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="rgb_array"
)
# # 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # EXPERIMENT: Change this to COMPLEX_MOVEMENT
# # 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# # 4. Wrap inside the Dummy Environment

env = DummyVecEnv([lambda: env])
# env = SubprocVecEnv([lambda: env], start_method="spawn") # EXPERIMENT: Try to change how we run the simulations


# # 5. Stack the frames
env = VecFrameStack(env, 4, channels_order="last")

# # 6. Normalize the observation/rewards/both
#Both
env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-08) # EXPERIMENT: See if this normalization (on observation, or on rewards with norm_rewards=True) changes performance of Agent
#Rewards only
#env = VecNormalize(env, training=True, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-08) # EXPERIMENT: See if this normalization (on observation, or on rewards with norm_rewards=True) changes performance of Agent
#Observation only 
#env = VecNormalize(env, training=True, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-08) # EXPERIMENT: See if this normalization (on observation, or on rewards with norm_rewards=True) changes performance of Agent


# #3.TRAIN THE MODEL
# # Import os for file path management

# # Callbacks

class TrainAndLoggingCallback(BaseCallback):
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

# Linear learning rate schedule
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"
# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# This is the AI model started
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=linear_schedule(3e-4),
    n_steps=2056,
) 

# Train the AI model, this is where the AI model starts to learn
import time
t_start = time.time()

model.learn(
    total_timesteps=100000, callback=callback
)  # EXPERIMENT: train for longer periods of time to see how it improves over that time span

t_elapsed = time.time() - t_start
print('Wall time: {} s'.format(round(t_elapsed, 2)))

# Save model

model_path = "./model_zip/"
model.save(model_path)
