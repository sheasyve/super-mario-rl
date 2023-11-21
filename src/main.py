import gym_super_mario_bros
import os
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (DummyVecEnv,SubprocVecEnv,VecFrameStack,VecNormalize)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
CHECKPOINT_DIR,LOG_DIR = "./train/", "./logs/"

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

def env_setup():
    #Set up Environment
    env = gym_super_mario_bros.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="rgb_array") # render_mode="human"
    #env = MonitorWrapper(env)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeObservation(env, 64)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-08)
    return env

def policy(env,callback, filename="archive/temp.model", learn=False, learning_rate=0.0001):
    #Set up and
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=learning_rate,# EXPERIMENT:Adjust this learning rate. So far we have found .0001 to be best.
        n_steps=512)
    if learn:
        model.learn(total_timesteps=50000, callback=callback)#Train the AI model. EXPERIMENT:train for longer periods of time. We have found it is best to use at least 50k.
    model.save(filename)
    return model

def train(model_name, env, callback, iterations_per_training_interval=3, training_iterations=5, test_steps=512*3):
    #Trains a given model for iterations_per_training_interval itrerations, then tests the model and records the average reward.
    #Repeats this process training_iterations times. 
    rewards = []
    for i in range(training_iterations):
        # Load the model
        model = PPO.load("./" + model_name, env=env)
        model.learn(total_timesteps=512 * iterations_per_training_interval, callback=callback) 
        model.save(model_name)
        
        # Test the model and record the average reward
        total_rewards, i = 0, 0
        state = env.reset()
        while i < test_steps:
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)   
            total_rewards += reward
                
            env.render()
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
    #plt.savefig("./Figures/" + model_name)
    plt.show()
    
def main():
    # Set up environment
    env = env_setup() 
    #view_initial_images(env)
    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    # Test parameters
    fn = "archive/testmodel1"
    training_iterations, iterations_per_training_interval = 100, 5
    learning_rate = 0.0001

    # Create (and save) the initial model. Change "learn" to "True" if you want to skip the plotting step and train it normally
    model = policy(env,callback, filename=fn, learn=False, learning_rate=learning_rate)

    # Train, and obtain average rewards as iterations increase. Then plot these rewards
    rewards = train(fn, env, callback, training_iterations=training_iterations, iterations_per_training_interval=iterations_per_training_interval)
    plot(rewards, iterations_per_training_interval, training_iterations)

    # Perform an extensive test after model has been fully trained
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)#Evaluate model to get training rewards
    #print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    
if __name__ == "__main__":
    main()
