import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()
print("Initial state:", obs)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("Next state:", obs)

env.close()