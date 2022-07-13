#from agents import evaluate
from ppo_agent import PPOAgent
import gym
import multiprocessing

if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    envs = gym.vector.make('CarRacing-v1', verbose=0, num_envs=cpu_count)

    agent = PPOAgent(envs, cpu_count)
    agent.train(episodes=2000, batch_size=512)

