import gym
import numpy as np

EPOCHS = 50000
SOLVED = 195
CONSECUTIVE_TIMESTEP = 100
MAX_CART_RANGE = 2.4
MAX_POLE_ANGLE = 15
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.3
STATES = 4
ACTIONS = 2
BIN_SIZE = 20
EPSILON = 0.4

class QLearning:
    def __init__(self) -> None:
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.table = [] 
        self.bins = [] 

    def initialize_QTable(self, bin_size = BIN_SIZE):
        self.bins = [np.linspace(-4.8,4.8,bin_size),
            np.linspace(-4, 4, bin_size),
            np.linspace(-0.418, 0.418, bin_size),
            np.linspace(-4, 4, bin_size)]

        self.table = np.random.uniform(low=-1, high=1, size=([BIN_SIZE] * STATES + [ACTIONS]))

    def discrete(self, state):
        index = []
        for i in range(len(state)):
            index.append(np.digitize(state[i], self.bins[i]) - 1)
        return tuple(index)
        
    def fit(self, epochs=EPOCHS, discount=DISCOUNT_FACTOR, learning_rate=LEARNING_RATE, timestep=CONSECUTIVE_TIMESTEP, epsilon=EPSILON):
        rewards, steps = 0, 0
        score, done = 0, False

        for epoch in range(1, epochs+1):

            steps += 1
            state = self.discrete(self.env.reset())

            while not done:
                if epoch % timestep == 0:
                    self.env.render()
                if np.random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.table[state])
                obs, reward, done, _ = self.env.step(action)
                newstate = self.discrete(obs)
                score += reward

                if not done:
                    maxfuture = np.max(self.table[newstate])
                    q = self.table[state + (action, )]
                    newq = (1 - learning_rate) * q + learning_rate * (reward + discount * maxfuture)
                    self.table[state + (action, )] = newq
                    state = newstate

                else:
                    rewards += score
                    if score > SOLVED and steps >= CONSECUTIVE_TIMESTEP:
                        print("Problem solved")
                        break
                    
            if epoch % timestep == 0:
                print(reward/timestep)


qlearn = QLearning()
qlearn.initialize_QTable()
qlearn.fit()

