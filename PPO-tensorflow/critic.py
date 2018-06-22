import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from utils import Config


class Critic:
    def __init__(self):
        with tf.variable_scope('critic') as sc:
            self._build_graph()
            self.params = tf.trainable_variables(sc.name)
            self._build_loss()
            self._build_train_op()

    def _build_graph(self):
        self.states = tf.placeholder(tf.float32, [None, Config.data.state_dim], 'state')
        hidden1 = Config.data.state_dim * 10
        hidden3 = 5
        hidden2 = int(np.sqrt(hidden1 * hidden3))

        net = slim.fully_connected(self.states, hidden1)
        net = slim.fully_connected(net, hidden2)
        net = slim.fully_connected(net, hidden3)
        self.value = slim.fully_connected(net, 1, activation_fn=None)

    def _build_loss(self):
        self.target_v = tf.placeholder(tf.float32, [None, 1], 'target_v')
        self.loss = tf.reduce_mean(tf.square(self.target_v - self.value))

    def _build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(Config.train.critic_lr).minimize(self.loss, var_list=self.params)
