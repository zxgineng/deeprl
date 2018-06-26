import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from utils import Config



class Actor:
    def __init__(self):
        with tf.variable_scope('actor') as sc:
            self._build_graph()
            self.params = tf.trainable_variables(sc.name)
            self._build_loss()
            self._build_train_op()

    def _build_graph(self):
        self.states = tf.placeholder(tf.float32, [None, Config.data.state_dim], 'states')
        # all layers use tanh in paper
        hidden1 = Config.data.state_dim * 10
        net = slim.fully_connected(self.states, hidden1)
        hidden2 = int(np.sqrt(Config.data.state_dim * Config.data.action_dim) * 10)
        hidden3 = Config.data.action_dim * 10
        net = slim.fully_connected(net, hidden2)
        net = slim.fully_connected(net, hidden3)
        self.mean = slim.fully_connected(net, Config.data.action_dim, None)
        log_stdd = tf.get_variable('log_stdd', [1, Config.data.action_dim], tf.float32,
                                   tf.constant_initializer(0.0))
        self.stdd = tf.exp(log_stdd)
        self.policy = tf.distributions.Normal(loc=self.mean, scale=self.stdd)
        self.sample = tf.squeeze(self.policy.sample(1), axis=[0, 1])

    def _build_loss(self):
        with tf.variable_scope('loss'):

            self.old_mean = tf.placeholder(tf.float32, [None, Config.data.action_dim], 'old_mean')
            self.old_stdd = tf.placeholder(tf.float32, [None, Config.data.action_dim], 'old_stdd')
            self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
            self.actions = tf.placeholder(tf.float32, [None, Config.data.action_dim], name='actions')

            old_policy = tf.distributions.Normal(self.old_mean, self.old_stdd)
            ratio = self.policy.prob(self.actions) / old_policy.prob(self.actions)
            if Config.train.surrogate_clip:
                clipped_ratio = tf.clip_by_value(ratio, 1 - Config.train.clip_epsilon,
                                                 1 + Config.train.clip_epsilon)
                surrogate_loss = tf.minimum(self.advantage * ratio,
                                            self.advantage * clipped_ratio)
                self.loss = -tf.reduce_mean(surrogate_loss)

            else:
                self.kl = tf.reduce_mean(tf.distributions.kl_divergence(old_policy, self.policy))  # [N,a_dim]
                loss1 = tf.reduce_mean(ratio * self.advantage)
                loss2 = Config.train.kl_loss_lam * self.kl
                loss3 = Config.train.kl_loss_eta * tf.square(
                    tf.maximum(0.0, self.kl - 2.0 * Config.train.kl_target))

                self.loss = -(loss1 - loss2 - loss3)

    def _build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(Config.train.actor_lr).minimize(self.loss, var_list=self.params)
