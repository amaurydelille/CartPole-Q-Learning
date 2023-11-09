import random
import math
import gym
import numpy as np

EPOCH = 10000
DISCOUNT = 0.7 
LEARNING_RATE = 0.1
RENDER = 100

env = gym.make('CartPole-v1', render_mode="human") 

states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10
for i in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = random.randint(0, 1)
        n_step, reward, done, info = env.step(action)
        score += reward
        env.render()

    print(f"Episode : {i} - Score : {score}")

env.close()
