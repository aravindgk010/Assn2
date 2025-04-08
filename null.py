import time
import gymnasium as gym
import aisd_examples

# Create the environment
env = gym.make("aisd_examples/RedBall-V0", render_mode="human")

# Reset the environment
observation= env.reset()


# Run the agent for 100 steps
for step in range(1000):
    action = env.action_space.sample()  # Random action selection
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished, resetting environment...")
        observation, info = env.reset()

env.close()
