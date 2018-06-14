import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from architecture import Graph
from hooks import TrainHook


class Agent:
    def __init__(self,env):
        self.env = env
        self.ep_state = []
        self.ep_action = []
        self.ep_reward = []
        self.sess = None
        self._config_initialize()

    def _config_initialize(self):
        """initialize env config"""
        if 'n' not in vars(self.env.action_space):
            Config.data.action_dim = self.env.action_space.shape[0]
            Config.data.action_bound = self.env.action_space.high[0]
            Config.data.action_type = 'continuous'
        else:
            Config.data.action_num = self.env.action_space.n
            Config.data.action_type = 'discrete'
        Config.data.state_dim = self.env.observation_space.shape[0]

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.states = features
        self.loss, self.train_op, self.predictions, self.training_hooks, self.evaluation_hooks = None, None, None, None, None
        self.build_graph()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks,
            evaluation_hooks=self.evaluation_hooks)

    def build_graph(self):
        graph = Graph()
        logits = graph.build(self.states)
        tf.nn.softmax(logits,-1,'action_prob')

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits)
            self._build_train_op()

    def _build_loss(self,logits):
        action = tf.placeholder(tf.int32, [None], 'action')
        vt = tf.placeholder(tf.float32, [None],'action_value')
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action,logits=logits) * vt)

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, self.global_step,
            optimizer=tf.train.AdamOptimizer(Config.train.learning_rate),
            learning_rate=Config.train.learning_rate,
            name="train_op")
        self.training_hooks = [TrainHook(self)]

    def choose_action(self, observation, one_hot=False):

        observation = np.expand_dims(observation, 0)
        prob = self.sess.run('action_prob:0', {'current_state:0': observation})
        action = np.random.choice(range(prob.shape[1]), p=prob.ravel())

        if one_hot:
            action_index = action
            action = np.zeros(Config.data.action_num)
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
