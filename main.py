import random
import math
import gym
import numpy as np

EPOCHS = 10000
DISCOUNT = 0.7 
LEARNING_RATE = 0.1
RENDER = 100
PROGRESS = 100
PLUS = 50
MINUS = -50
epsilon = 1  
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPOCHS // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

class QLearning:
    def __init__(self) -> None:
        self.env = gym.make('CartPole-v1', new_step_api=True, render_mode="human")
        self.QTable = []
        self.bins = []
        self.space = len(self.env.observation_space.high)
    def random_init(self): 
        states = self.env.observation_space.shape[0]
        actions = self.env.action_space.n

        episodes = 10
        for i in range(1, episodes+1):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                action = random.randint(0, 1)
                n_step, reward, done, info = self.env.step(action)
                score += reward
                self.env.render()

            print(f"Episode : {i} - Score : {score}")

        self.env.close()

    def initialize_QTable(self):
        self.bins = [] 
        self.bins.append([4 * (-4.8 + i * 4.8) for i in range(21)]) #cart position
        self.bins.append([4 * (-4 + i * 4) for i in range(21)]) #cart velocity
        self.bins.append([4 * (-0.418 + i * 0.418) for i in range(21)]) #pole angle
        self.bins.append([4 * (-4 + i * 4) for i in range(21)]) #pole angular velocity

        self.QTable = np.random.uniform(-2, 0, [len(self.bins[0])] * self.space + [self.env.action_space.n])
        self.QTable.shape

    def state_to_discrete(self, state, bins, space):
        state_i = []
        for i in range(self.space):
            state_i.append(np.digitize(state[i], self.bins[i]) - 1)
        return tuple(state_i)
        

        
qlearn = QLearning()
qlearn.initialize_QTable()
scores = []
renderer = {"epoch": [], "average": [], "min": [], "max": []}

for epoch in range(EPOCHS):
    discrete = qlearn.state_to_discrete(qlearn.env.reset(), qlearn.bins, qlearn.space)

    done = False
    count = 0

    while not done:
        if epoch % PROGRESS == 0:
            qlearn.env.render()

        count += 1
        if np.random.random() > epsilon:
            action = np.random.randint(qlearn.QTable[discrete])
        else:
            action = np.random.randint(0, qlearn.env.action_space.n)
        new_state, reward, terminated, truncated, info = qlearn.env.step(action)

        newdiscrete = qlearn.state_to_discrete(new_state, qlearn.bins, qlearn.space)

        maxQprime = np.max(qlearn.QTable[newdiscrete])
        Q = qlearn.QTable[discrete + (action, )]

        if done and count < 200:
            reward = MINUS

        qlearn.QTable[discrete + (action, )] = (1 - LEARNING_RATE) * Q + LEARNING_RATE * (reward + DISCOUNT * maxQprime) #Bellmann
        discrete = newdiscrete
        qlearn.env.render()
    scores.append(count)

qlearn.env.close()
