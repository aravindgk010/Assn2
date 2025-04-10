import gymnasium as gym #type: ignore
import aisd_examples
import matplotlib.pyplot as plt

from stable_baselines3 import DQN #type: ignore

env = gym.make("aisd_examples/RedBall-V0", render_mode="human")

model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_blocks")

del model   #remove to demonstrate saving and loading

model = DQN.load("dqn_blocks")

obs, info = env.reset()

episode_rewards = []
# Run for 10 episodes
for episode in range(10):
    done = False
    total_reward = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward+= reward

        if terminated or truncated:
            print(f"Episode {episode + 1} - Total Reward: {total_reward}")
            episode_rewards.append(total_reward)
            obs, info = env.reset()
            done = True

env.close()

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards, marker='o', linestyle='-')
plt.title('DQN - Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.tight_layout()
plt.savefig("dqn_rewards.png")
plt.show()