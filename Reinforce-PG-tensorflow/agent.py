import numpy as np
import tensorflow as tf

from utils import Config


class Agent:
    def __init__(self):
        self.ep_state = []
        self.ep_action = []
        self.ep_reward = []
        self.sess = None

    def choose_action(self, observation, one_hot=False):

        observation = np.expand_dims(observation, 0)
        prob = self.sess.run('action_prob:0', {'current_state:0': observation})
        action = np.random.choice(range(prob.shape[1]), p=prob.ravel())

        if one_hot:
            action_index = action
            action = np.zeros(Config.data.num_action)
            action[action_index] = 1
        return action

    def store_transition(self, state, action, reward):
        self.ep_state.append(state)
        self.ep_action.append(action)
        self.ep_reward.append(reward)

    def _discount_and_norm_rewards(self):
        discounted_ep_reward = np.zeros_like(self.ep_reward)
        running_add = 0
        for t in reversed(range(len(self.ep_reward))):
            running_add = running_add * Config.train.reward_decay + self.ep_reward[t]
            discounted_ep_reward[t] = running_add

        # normalize episode rewards
        discounted_ep_reward -= np.mean(discounted_ep_reward)
        discounted_ep_reward /= np.std(discounted_ep_reward)

        return discounted_ep_reward

    def learn(self):
        discounted_ep_reward = self._discount_and_norm_rewards()

        return tf.train.SessionRunArgs('global_step:0',
                                       feed_dict={'action_value:0': discounted_ep_reward,
                                                  'current_state:0': np.vstack(self.ep_state),
                                                  'action:0': self.ep_action})

    def ep_reset(self):
        self.ep_state = []
        self.ep_action = []
        self.ep_reward = []
