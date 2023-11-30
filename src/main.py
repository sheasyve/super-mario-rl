import random
import gym_super_mario_bros
import os
import argparse

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (DummyVecEnv,SubprocVecEnv,VecFrameStack,VecNormalize)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed


CHECKPOINT_DIR,LOG_DIR = "./train/", "./logs/"
# Set render_mode to 'human' to display the model on screen as it learns
# Set render_mode to 'rgb_array' for large increase in training speed (40% in my case)
# render_mode = 'human' 
render_mode = 'rgb_array' 
processes = 1 #How many threads. 1 for single threaded.

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

def view_initial_images(env):
    #Draw the initial preprocessed screen (first 4 stacked frames)
    state = env.reset()
    state, reward, done, info = env.step([5])
    plt.figure(figsize=(20, 16))
    for idx in range(state.shape[3]):
        plt.subplot(1,4,idx+1)
        plt.imshow(state[0][:,:,idx])
    plt.show()

def make_env(stages,random,rank=0,seed = 0):
    set_random_seed(seed)
    if random:
        env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0", apply_api_compatibility=True, render_mode=render_mode)
    else:
        env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0", apply_api_compatibility=True, render_mode=render_mode, stages=stages)
    #env = MonitorWrapper(env)
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeObservation(env, 64)
    env = GrayScaleObservation(env, keep_dim=True)
    if processes != 1:
        env.reset(seed=seed + rank)
    return env

def env_setup(stages, random=False):
    #Set up Environment
    if processes == 1:#Single thread
        env = make_env(stages,random)
        env = DummyVecEnv([lambda: env])
    else:#Multithread
        env = SubprocVecEnv(
            [make_env(stages,random,i) for i in range(processes)],
            start_method="fork",
        )
    env = VecFrameStack(env, 4, channels_order="last")
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-08)
    return env

def policy(env,callback, filename="archive/temp.model", learn=False, learning_rate=0.0001):
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=learning_rate,# EXPERIMENT:Adjust this learning rate. So far we have found .0001 to be best.
        n_steps=512)
    if learn:
        model.learn(total_timesteps=50000, callback=callback)#Train the AI model. EXPERIMENT:train for longer periods of time. We have found it is best to use at least 50k.
    model.save(filename)
    return model

def train(model_name, env, callback, iterations_per_training_interval=3, training_iterations=5, test_steps=512*3):
    # Trains a given model for iterations_per_training_interval itrerations, then tests the model and records the average reward.
    # Repeats this process training_iterations times. 
    rewards = []
    for i in range(training_iterations):
        # Load the model
        model = PPO.load("./" + model_name, env=env)
        model.learn(total_timesteps=512 * iterations_per_training_interval, callback=callback, progress_bar=True) 
        model.save(model_name)
        
        # Test the model and record the average reward
        total_rewards, i = 0, 0
        state = env.reset()
        while i < test_steps:
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)   
            total_rewards += reward
            i += 1
        avg_reward = (total_rewards/test_steps)[0]
        rewards.append(avg_reward)
        print(f"avg reward: {avg_reward}")
    return rewards
    
def plot(rewards, iterations_per_training_interval, training_iterations):
    x_ticks = list(range(0, training_iterations))
    fig, ax = plt.subplots()
    ax.plot(x_ticks, rewards)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.title(f"Average reward over {training_iterations  * iterations_per_training_interval} iterations")
    plt.xlabel("Iteration number", color="black", size=28)  
    plt.ylabel("Average reward", color="black", size=28)
    plt.plot(x_ticks, rewards, marker='o')
    # plt.savefig("./Figures/" + model_name)
    plt.show()
    
def final_test(fn,env):
    #Renders the finished model playing a game of mario
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
    parser = argparse.ArgumentParser(description='Train RL models for Super Mario')
    parser.add_argument('-m', '--mode', type=str, default='continious', metavar='', help='continious | iterative')
    parser.add_argument('-f', '--filename', type=str, metavar='', default='archive/templmodel', help='Filename to begin training model with. Recommended for continious mode.')
    parser.add_argument('-r', '--random', action='store_true', help='Train the model with stage 1-1 and 4 random level combinations. From world 1-8 and stage 1-4')
    parser.add_argument('-tr', '--truerandom', action='store_true', help='Train the model with a random stage at every death and reset')
    args = parser.parse_args()
    return args

def stageing(args):
    stages = ['1-1']
    if args.random:
        for i in range(4):
            world = random.randint(1, 8)
            stage = random.randint(1, 4)
            stage_string = f'{world}-{stage}'
            stages.append(stage_string)
    return stages

def main():
    args = parsing()
    stages = stageing(args)
        
    # Set up environment
    env = env_setup(stages=stages, random=args.truerandom) 
    #view_initial_images(env)
    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    # Test parameters
    fn = args.filename
    training_iterations, iterations_per_training_interval = 100, 5
    learning_rate = 0.0001

    # Create (and save) the initial model. Change "learn" to "True" if you want to skip the plotting step and train it normally
    model = policy(env,callback, filename=fn, learn=False, learning_rate=learning_rate)

    # Train, and obtain average rewards as iterations increase. Then plot these rewards
    rewards = train(fn, env, callback, training_iterations=training_iterations,
                     iterations_per_training_interval=iterations_per_training_interval)
    plot(rewards, iterations_per_training_interval, training_iterations)
    # Perform an extensive test after model has been fully trained
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)#Evaluate model to get training rewards
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    
    #final_test(fn,env)
    
    
if __name__ == "__main__":
    main()

    
