import gymnasium as gym #type: ignore
import aisd_examples
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# create environment
env = gym.make("aisd_examples/RedBall-V0", render_mode="human")

# Define Q-table dimensions
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.random.rand(num_states, num_actions)

""" final  hyperparameter values"""
# hyperparameters - 1
episodes = 50
alpha = 0.5       # High learning rate (keeps fast updates)
gamma = 0.95      # Slightly increases long-term reward consideration
epsilon = 0.08    # Controlled exploration
decay = 0.995     # Slightly lower decay to balance exploration-exploitation #0.12

# Metrics for plotting
episode_rewards = []
steps_per_episode = []

# training loop
for i in range(episodes):
    state_dict = env.reset()
    state = state_dict['position']

    total_reward = 0
    steps = 0
    done = False


    while not done:
        os.system('cls' if os.name == 'nt' else 'clear')    #'cls' if os.name == 'nt' else 'clear' - new change original 'cls' # cross-platform console clear
        
        print("episodes #", i+1, "/", episodes)
        
        env.render()
        time.sleep(0.05)

        steps += 1

        # select action using epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        
        else:
            action = np.argmax(q_table[state]) # choose action with highest Q-value

        # take action
        next_state_dict, reward, terminated, truncated, info = env.step(action)
        next_state = next_state_dict['position']
        
        # q-learning update rule
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

        # update state
        state = next_state

        total_reward += reward
        
        done = terminated or truncated
    
    # decay exploration rate
    epsilon = max(0.01, epsilon*(1-decay))

    # Logging metrics
    episode_rewards.append(total_reward)
    steps_per_episode.append(steps)

    print(f"Episode {i+1}: Steps = {steps}, Total Reward = {total_reward}")
    time.sleep(2)

# Close environment after training
env.close()

# Plotting the results
plt.figure(figsize=(12, 5))

# Plot for rewards
plt.plot(episode_rewards, label="Total Rewards")
plt.title("Episode Rewards - Qlearning")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.tight_layout()
plt.savefig("media/q_learning_rewards.png")
plt.show()
