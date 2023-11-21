import gym_super_mario_bros
import os
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from matplotlib import pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (DummyVecEnv,SubprocVecEnv,VecFrameStack,VecNormalize)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
CHECKPOINT_DIR,LOG_DIR = "./train/", "./logs/"

def plot(state):
    #Not sure what this is plotting
    plt.figure(figsize=(20, 16))
    for idx in range(state.shape[3]):
        plt.subplot(1,4,idx+1)
        plt.imshow(state[0][:,:,idx])
    plt.show()

class TrainAndLoggingCallback(BaseCallback):
    #Setup model saving callback
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

def env_setup():
    #Set up Environment
    env = gym_super_mario_bros.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="rgb_array")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])#env = SubprocVecEnv([lambda: env], start_method="spawn") # EXPERIMENT: Try to change how we run the simulations by using multiprocessing
    env = VecFrameStack(env, 4, channels_order="last")
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-08)# 6.Normalize the observation and rewards.
    return env

def policy(env,callback):
    #Set up and start model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=0.00001,# EXPERIMENT:Adjust this learning rate. So far we have found .0001 to be best.
        n_steps=512)
    model.learn(total_timesteps=50000, callback=callback)#Train the AI model. EXPERIMENT:train for longer periods of time. We have found it is best to use at least 50k.
    model.save("archive/temp.model")
    return model
    
def main():
    env = env_setup()
    state = env.reset()#Draw the initial preprocessed screen
    state, reward, done, info = env.step([5])
    plot(state)
    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
    model = policy(env,callback)#Setup and run model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)#Evaluate model to get training rewards
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    
if __name__ == "__main__":
    main()
