import datetime
import tensorflow as tf
from tensorflow_probability.python.distributions.beta import Beta
import gym
import numpy as np
from pathlib import Path
from model import create_model_actor, create_model_critic

class PPOAgent():

    def __init__(self, envs, num_envs):
        self.envs = envs
        self.num_envs = num_envs
        self.actor_model = create_model_actor()
        self.critic_model = create_model_critic()

    def get_action(self, observation, action_space):
        observation = np.expand_dims(observation, axis=0)
        pred_actor = self.actor_model.predict(observation)[0]
        pred_critic = self.critic_model.predict(observation)[0]
        distribution = Beta(pred_actor, pred_critic)
        action = distribution.sample().numpy()
        action[0] = np.interp(action[0], [0, 1], [-1, 1])
        return action

    def train(self, episodes = 1000, log_interval = 10, model_dir = 'models', save_interval = 5, buffer_size = 2000, batch_size = 128, gamma = 0.99, ppo_epochs = 10, clip_epsilon = 0.1):
        model_dir = Path(model_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        model_dir.mkdir(parents=True, exist_ok=True)

        training_start_time = datetime.datetime.now()
        print('Started training at', training_start_time.strftime('%d-%m-%Y %H:%M:%S'))

        episode_rewards = []

        transitions = []

        for episode in range(1, episodes + 1):

            observations = self.envs.reset()
            episode_reward = 0

            done = False
            while not done:

                pred_actor = self.actor_model.predict(observations)

                distribution = Beta(pred_actor[:,:,0], pred_actor[:,:,1])
                actions = distribution.sample()

                log_probs = tf.reduce_sum(distribution.log_prob(actions), axis=1)

                action_numpy = actions.numpy()
                action_numpy[:,0] = np.interp(action_numpy[:,0], [0, 1], [-1, 1])

                new_observations, rewards, dones, _ = self.envs.step(action_numpy)
                done = dones.all()
                
                episode_reward += rewards.sum()
                for i in range(self.num_envs):
                    if not dones[i]:
                        transitions.append((observations[i], actions[i], log_probs[i], rewards[i], new_observations[i]))

                if len(transitions) >= buffer_size:

                    states = tf.convert_to_tensor([x[0] for x in transitions])
                    actions = tf.convert_to_tensor([x[1] for x in transitions])
                    old_a_logp = tf.expand_dims(tf.convert_to_tensor([x[2] for x in transitions]), axis=1)
                    rewards = tf.expand_dims(tf.convert_to_tensor([x[3] for x in transitions], dtype=np.float32), axis=1)
                    new_states = tf.convert_to_tensor([x[4] for x in transitions])

                    discounted_rewards = rewards + gamma * self.critic_model(new_states)
                    adv = discounted_rewards - self.critic_model(states)

                    def gen_batches(indices, batch_size):
                        for i in range(0, len(indices), batch_size):
                            yield indices[i:i + batch_size]

                    for _ in range(ppo_epochs):
                        indices = np.arange(buffer_size)
                        np.random.shuffle(indices)

                        for batch in gen_batches(indices, batch_size):

                            with tf.GradientTape(persistent=True) as tape:

                                ab = self.actor_model(tf.gather(states, batch))[0]
                                alpha, beta = ab[:, 0], ab[:, 1]
                                dist = Beta(alpha, beta)
                                a_logp = tf.reduce_sum(dist.log_prob(tf.gather(actions, batch)), axis=1, keepdims=True)
                                ratio = tf.exp(a_logp - tf.gather(old_a_logp, batch))
                                surr1 = ratio * tf.gather(adv, batch)
                                surr2 = tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * tf.gather(adv, batch)
                                action_loss = tf.reduce_mean(-tf.minimum(surr1, surr2))

                                value_loss = tf.reduce_mean(
                                    tf.losses.mse(
                                        tf.gather(discounted_rewards, batch),
                                        self.critic_model(tf.gather(states, batch))
                                    ))

                                loss = action_loss + 2 * value_loss

                            g_actor = tape.gradient(loss, self.actor_model.trainable_variables)
                            self.actor_model.optimizer.apply_gradients(zip(g_actor, self.actor_model.trainable_variables))
                            g_critic = tape.gradient(loss, self.critic_model.trainable_variables)
                            self.critic_model.optimizer.apply_gradients(zip(g_critic, self.critic_model.trainable_variables))

                            del tape

                    transitions.clear()

                observations = new_observations

            episode_rewards.append(episode_reward)

            if not episode % log_interval:
                print(f'Episode {episode} | Reward: {episode_reward:.02f} | Moving Average: {np.average(episode_rewards[-50:]):.02f}')

            if not episode % save_interval:
                self.actor_model.save(model_dir / f'episode-{episode}-actor.h5')
                self.critic_model.save(model_dir / f'episode-{episode}-critic.h5')

        self.actor_model.save(model_dir / 'model-actor.h5')
        self.critic_model.save(model_dir / 'model-critic.h5')

        training_end_time = datetime.datetime.now()
        print('Finished training at', training_end_time.strftime('%d-%m-%Y %H:%M:%S'))
        print('Total training time:', training_end_time - training_start_time)
        np.savetxt(model_dir / 'rewards.txt', episode_rewards)