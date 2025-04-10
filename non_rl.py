import gymnasium as gym #type: ignore
import aisd_examples
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# Create environment
env = gym.make("aisd_examples/RedBall-V0", render_mode="human")

episodes = 50
episode_rewards = []
steps_per_episode = []

for i in range(episodes):
    state_dict = env.reset()
    state = state_dict["position"]

    done = False
    total_reward = 0
    steps = 0

    while not done:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Non-RL Agent | Episode:", i+1, "/", episodes)

        env.render()
        time.sleep(0.05)

        steps += 1

        # RULE-BASED ACTION SELECTION
        if state in [0]:           # Ball far left
            action = 0             # Rotate left
        elif state in [3]:         # Ball far right
            action = 2             # Rotate right
        else:                      # Ball centered (zones 1 or 2)
            action = 1             # Stay still

        # Step through environment
        next_state_dict, reward, terminated, truncated, info = env.step(action)
        state = next_state_dict["position"]

        total_reward += reward
        done = terminated or truncated

    print(f"Episode {i+1}: Steps = {steps}, Total Reward = {total_reward}")
    episode_rewards.append(total_reward)
    steps_per_episode.append(steps)

    time.sleep(1)

env.close()

# Save plot
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Total Rewards per Episode")
plt.title("Non-RL Agent Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("non_rl_rewards.png")
plt.show()




























































































































































