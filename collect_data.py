import gymnasium as gym
import numpy as np
import pickle

env = gym.make("CarRacing-v3", render_mode="rgb_array")

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")


dataset = []
num_episodes = 100

for episode in range(num_episodes):

    obs, _ = env.reset()
    
    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": []
    }

    done = False
    truncated = False

    while not (done or truncated):

        action = env.action_space.sample()

        next_obs, reward, done, truncated, info = env.step(action)

        episode_data["observations"].append(obs)
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["dones"].append(done)

        obs = next_obs

    dataset.append(episode_data)

env.close()

with open("carRacing_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("Episodes collected:", len(dataset))
print("Example episode length:", len(dataset[0]["observations"]))