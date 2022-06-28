from agent import Agent
import numpy as np
import gym
from gym import wrappers
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

def training(episodes, save_frequency, gamma, epsilon, lr, mem_size, batch_size, epsilon_end, model_id, epsilon_dec):
    with tf.device('GPU'):
        env = gym.make('LunarLander-v2')
        agent = Agent(env, lr, gamma, epsilon, mem_size, batch_size, epsilon_end, model_id, epsilon_dec)

        begin_episode = agent.load_model()
        scores = []
        eps_history = []
        
        for episode in range(begin_episode, episodes+1):
            done = False
            score = 0
            observation = env.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, observation_, int(done))
                observation = observation_
                agent.learn()

            eps_history.append(agent.epsilon)
            scores.append(score)

            avg_score = np.mean(scores[max(0, (episode-begin_episode)-50):((episode-begin_episode)+1)])
            print('episode: ', episode,'score: %.2f' % score,
                ' average score %.2f' % avg_score)

            if episode % save_frequency == 0 and episode > 0:
                agent.save_model(episode)

def test_visualize(gamma, epsilon, lr, mem_size, batch_size, epsilon_end, model_id, epsilon_dec):
    env = gym.make('LunarLander-v2')

    agent = Agent(env, lr, gamma, epsilon, mem_size, batch_size, epsilon_end, model_id, epsilon_dec)
    agent.load_model()
    
    q_eval = agent.q_eval
    for i in range(10):
        done = False
        observation = env.reset()
        while not done:
            actions = q_eval.predict(observation[np.newaxis, :], verbose= 0)
            action = np.argmax(actions)
            observation, reward, done, info = env.step(action)
            env.render()

if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == 'train':
        training(int(args[1]), int(args[2]), float(args[3]), float(args[4]), float(args[5]), int(args[6]), int(args[7]), float(args[8]), int(args[9]), float(args[10]))
    elif args[0] == 'test':
        test_visualize(float(args[3]), float(args[4]), float(args[5]), int(args[6]), int(args[7]), float(args[8]), int(args[9]), float(args[10]))
