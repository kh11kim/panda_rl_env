import numpy as np
import gym
import panda_rl_env

env = gym.make("PandaReach2-v0",render=True)
for i in range(100):
    obs = env.reset()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
input()