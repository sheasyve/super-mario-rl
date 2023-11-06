# 1.SETUP MARIO
# Import the game
import gym_super_mario_bros

# 2.PREPROCESS ENVIRONMENT
# Install pytorch
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation

# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace

# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Setup game

# env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
# # Create a flag - restart or not
# done = True

# for step in range(100000):
# # Start the game to begin with
#     if done:
#         # Start the game
#         env.reset()
#     # Do actions
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     # Show the game on the screen
#     done = terminated or truncated
#     env.render()
# # Close the game
# env.close()


# 1. Create the base environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# # 4. Wrap inside the Dummy Environment

# 3.TRAIN THE MODEL
# Import os for file path management
import os

# Import PPO for algos
from stable_baselines3 import PPO

# # Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback


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


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"
# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# This is the AI model started
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.000001,  # EXPERIMENT: Adjust this learning rate
    n_steps=512,
)  # EXPERIMENT: Try using 'MlpPolicy' or "MultiInputPolicy" instead
# Train the AI model, this is where the AI model starts to learn
model.learn(
    total_timesteps=10000, callback=callback
)  # EXPERIMENT: train for longer periods of time to see how it improves over that time span
model.save("test_baseline")


# 4.TEST THE MODEL

# Load model
# model = PPO.load('./test_name')
# state = env.reset()
# # Start the game
# # state = env.reset()
# # Loop through the game


# while True:

#     action, _ = model.predict(state)
#     state, reward, done, info = env.step(action)

#     env.render()
