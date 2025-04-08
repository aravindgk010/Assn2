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
episodes = 100
alpha = 1.0       # High learning rate (keeps fast updates)
gamma = 0.2       # Slightly increases long-term reward consideration
epsilon = 0.08    # Controlled exploration
decay = 0.12      # Slightly lower decay to balance exploration-exploitation

# Metrics for plotting
episode_rewards = []
steps_per_episode = []

# training loop
for i in range(episodes):
    state, info = env.reset()

    """ new change"""
    # Ensure state is an integer (avoid potential indexing issues)
    if isinstance(state, tuple):
        state = state[0]
        #state = (state,) # new change
        #state = state.get("State", 0)   # default to 0 if key doesn't exist

    total_reward = 0
    steps = 0
    done = False

    #Track visited states
    visited_states = set()

    while not done:
        os.system('cls' if os.name == 'nt' else 'clear')    #'cls' if os.name == 'nt' else 'clear' - new change original 'cls' # cross-platform console clear
        
        print("episodes #", i+1, "/", episodes)
        
        env.render()
        time.sleep(0.05)

        # select action using epsilon-greedy policy
        if np.random.uniform() < epsilon:
            #action = np.random.choice([0,1, 2, 3]) # random action
            """ new change old one above"""
            #action = np.random.choice(env.action_space.n)  # Use environment action space
            action = env.action_space.sample()
        
        else:
            action = np.argmax(q_table[state]) # choose action with highest Q-value

        # take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Ensure next_state is integer
        if isinstance(next_state, dict):
            next_state = next_state.get("State", 0)

        # Done flag
        done = terminated or truncated

        # Modified reward function
        step_penalty = -1  # Penalize each step
        revisit_penalty = -5 # Penalty for revisiting the same state
        goal_reward = 100

        if done:
            reward+= goal_reward  # If goal is reached

        # penalize id revisiting the same state
        if state in visited_states:
            reward += revisit_penalty
        else:
            visited_states.add(state)
        
        # q-learning update rule
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

        # update state
        state = next_state
        total_reward += reward
        steps += 1 
    
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
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label="Total Rewards")
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()

# Plot for steps per episode
plt.subplot(1, 2, 2)
plt.plot(steps_per_episode, label="Steps per Episode", color="red")
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()

plt.tight_layout()
plt.show()
