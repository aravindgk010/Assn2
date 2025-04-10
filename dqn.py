import os
import gymnasium as gym
import aisd_examples
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

# Wrap environment with Monitor
env = Monitor(gym.make("aisd_examples/RedBall-V0", render_mode="human"))

# Set up EvalCallback to evaluate and log best model
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs_dqn/",
    log_path="./logs_dqn/",
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Create and train the model
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, callback=eval_callback)
model.save("dqn_redball")

# Close environment
env.close()

# Load and plot evaluation results
data = np.load("./logs_dqn/evaluations.npz")
timesteps = data["timesteps"]
episode_rewards = data["results"].mean(axis=1)

plt.figure(figsize=(12, 5))
plt.plot(timesteps, episode_rewards, marker='o', linestyle='-')
plt.title("Episode Rewards - DQN")
plt.xlabel("Timesteps")
plt.ylabel("Average Reward")
plt.tight_layout()
plt.savefig("dqn_rewards.png")
plt.show()
