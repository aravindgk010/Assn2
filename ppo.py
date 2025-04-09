import gymnasium as gym #type: ignore
import aisd_examples
import matplotlib.pyplot as plt

from stable_baselines3 import PPO #type: ignore

env = gym.make("aisd_examples/RedBall-V0", render_mode="human")

# Initialize the PPO model
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("ppo_blocks")

# Clear the model from memory (for demonstration)
del model

# Load the saved model
model = PPO.load("ppo_blocks")

# Test the trained agent
obs, info = env.reset()

episode_rewards = []
# Run for 10 episodes
for episode in range(10):
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Check if the episode has ended
        if terminated or truncated:
            print(f"Episode {episode + 1} - Total Reward: {total_reward}")
            episode_rewards.append(total_reward)
            obs, info = env.reset()
            done = True

# Close the environment
env.close()

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards, marker='o', linestyle='-')
plt.title('DQN - Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid()
plt.show()
