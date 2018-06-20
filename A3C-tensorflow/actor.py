import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from utils import Config


class Actor:
    def __init__(self):
        with tf.variable_scope('actor') as sc:
            self._build_graph()
            self._build_loss()
            # self._build_train_op()
            self._build_grads()
            self.params = tf.trainable_variables(sc.name)

    def _build_graph(self):
        self.states = tf.placeholder(tf.float32, [None, Config.data.state_dim], 'states')

        hidden1 = Config.data.state_dim * 10
        net = slim.fully_connected(self.states, hidden1)
        if Config.data.action_type == 'discrete':
            hidden2 = int(np.sqrt(Config.data.state_dim * Config.data.action_num) * 10)
            hidden3 = Config.data.action_num * 10
            net = slim.fully_connected(net, hidden2)
            net = slim.fully_connected(net, hidden3,)
            logits = slim.fully_connected(net, Config.data.action_num, activation_fn=None)
            self.policy = tf.nn.softmax(logits)
        else:
            hidden2 = int(np.sqrt(Config.data.state_dim * Config.data.action_dim) * 10)
            hidden3 = Config.data.action_dim * 10
            net = slim.fully_connected(net, hidden2)
            net = slim.fully_connected(net, hidden3)
            self.mean = slim.fully_connected(net, Config.data.action_dim, None)
            self.stdd = slim.fully_connected(net, Config.data.action_dim, tf.nn.softplus)
            # log_stdd = tf.get_variable('log_stdd', [1, Config.data.action_dim], tf.float32,
            #                            tf.constant_initializer(0.0))
            # self.stdd = tf.exp(log_stdd)
            self.policy = tf.distributions.Normal(loc=self.mean, scale=self.stdd)
            self.sample = tf.squeeze(self.policy.sample(1), axis=[0, 1])

    def _build_loss(self):
        with tf.variable_scope('loss'):
            self.td_error = tf.placeholder(tf.float32, [None, 1], 'td_error')
            if Config.data.action_type == 'discrete':
                self.actions = tf.placeholder(tf.float32, [None],name='actions')

                log_prob = tf.reduce_sum(
                    tf.log(self.policy) * tf.one_hot(self.actions, Config.data.action_num, dtype=tf.float32), axis=1,
                    keep_dims=True)
                exp_v = log_prob * self.td_error
                entropy = -tf.reduce_sum(self.policy * tf.log(self.policy),
                                         axis=1, keep_dims=True)
                exp_v = Config.train.entropy_beta * entropy + exp_v
                self.loss = tf.reduce_mean(-exp_v)
            else:

                self.actions = tf.placeholder(tf.float32, [None, Config.data.action_dim], name='actions')

                log_prob = self.policy.log_prob(self.actions)
                exp_v = log_prob * self.td_error
                entropy = self.policy.entropy()
                exp_v = Config.train.entropy_beta * entropy + exp_v
                self.loss = tf.reduce_mean(-exp_v)

    def _build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(Config.train.actor_lr).minimize(self.loss,var_list=self.params)

    def _build_grads(self):
        self.grads = tf.gradients(self.loss, self.params)
