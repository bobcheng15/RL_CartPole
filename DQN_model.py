import tensorflow as tf
import random
import numpy as np


BATCH_SIZE = 20
MEM_MAX    = 100
GAMMA      = 0.9
EP_DECAY   = 0.95
EP_MIN     = 0.1
EP_INIT    = 1.0

class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.epsilon = EP_INIT;

        self.epsilon = EP_INIT;
        self.replay_memory = []

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(tf.keras.layers.Dense(24, activation="relu"))
        self.model.add(tf.keras.layers.Dense(action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        self.target_model =  tf.keras.models.Sequential()
        self.target_model.add(tf.keras.layers.Dense(24, input_shape=(observation_space,), activation="relu"))
        self.target_model.add(tf.keras.layers.Dense(24, activation="relu"))
        self.target_model.add(tf.keras.layers.Dense(action_space, activation="linear"))
        self.target_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))


    def add_to_replay(self, information):
        if (self.replay_memory.size < MEM_MAX):
            self.replay_memory.append(information)
        else:
            self.replay_memory.pop(0)
            self.replay_memory.append(information)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(action_space)
        else:
            q = self.model.predict(state)
            return np.argmax(q[0])
    def update_model(self):
        replay_data = random.sample(sefl.replay_memory, BATCH_SIZE)
        for state, action, reward, next_state, done in enumerate(replay_data):
            q_update = reward
            if not done:
                next_max_q = np.amax(self.target_model.predict(next_state)[0])
                q_update = reward + GAMMA * next_max_q
            apporx_q = self.model.predict(state)
            approx_q[0][action] = q_update
            self.modle.fit(state, q_value, verbose = 1)
        self.epsilon = max(self.epsilon * EP_DECAY, EP_MIN)
