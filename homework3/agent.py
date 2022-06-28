import numpy as np
from replay_buffer import ReplayBuffer
import tensorflowmodel as tfm
from tensorflow.keras.models import load_model
import os
import re

class Agent(object):
    def __init__(self, env, lr, gamma, epsilon,
                 mem_size, batch_size, epsilon_end,
                 model_id, epsilon_dec):
        self.action_space = [i for i in range(env.action_space.n)]
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_id = model_id
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.model_file = f"model-{model_id}-0-ep.h5"

        if not os.path.exists(env.spec.name):
            os.makedirs(env.spec.name)

        self.memory = ReplayBuffer(self.mem_size, env.observation_space.shape, env.action_space.n,
                                   discrete=True)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state, verbose= 0)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_eval = self.q_eval.predict(state, verbose=0)

            q_next = self.q_eval.predict(new_state, verbose=0)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                                  self.gamma*np.max(q_next, axis=1)*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min

    def create_model_dir(self):
        return f"{self.lr}-{self.gamma}-{self.epsilon}-{self.mem_size}-{self.batch_size}-{self.epsilon_min}-{self.epsilon_dec}"

    def get_params_from_dir(self, directory_name):
        split_param = directory_name.split("-")
        return int(split_param[0]), float(split_param[1]), float(split_param[2]), float(split_param[3]), float(split_param[4]), int(split_param[5])

    def save_model(self, episode):
        param_directory = self.create_model_dir()
        if not os.path.exists(os.path.join(self.env.spec.name, param_directory)):
            os.makedirs(os.path.join(self.env.spec.name, param_directory))
        self.model_file = os.path.join(self.env.spec.name, param_directory, f"model-{self.model_id}-{episode}-ep.h5")
        self.q_eval.save(self.model_file)

    def load_model(self):
        max_episode = 0
        act_dir = os.path.join(self.env.spec.name, self.create_model_dir())
        found_model = False
        for (dirpath, dirnames, filenames) in os.walk(act_dir):
            max_index = 0
            i = 0
            for filename in filenames:
                episode = int(filename.split('-')[2])
                if episode > max_episode and int(filename.split('-')[1]) == self.model_id:
                    max_index = i
                    max_episode = episode
                i = i+1
            self.model_file = filenames[max_index]
            self.q_eval = load_model(os.path.join(act_dir, self.model_file))
            found_model = True
            break
        if not found_model:
            self.q_eval = tfm.build_dqn(self.lr, self.env.action_space.n, self.env.observation_space.shape, 256, 256)
        return max_episode+1