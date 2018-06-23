import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from utils import Config


def _random_uniform_initializer(x):
    num = x.shape.as_list()[-1]
    return tf.random_uniform_initializer(
        minval=-1.0 / np.sqrt(num), maxval=1.0 / np.sqrt(num))


class Actor:
    def __init__(self):
        with tf.variable_scope('actor') as sc:
            self._build_graph()
            self.params = tf.trainable_variables(sc.name)

    def _build_graph(self):
        self.states = tf.placeholder(tf.float32, [None, Config.data.state_dim], 'states')
        net = slim.fully_connected(self.states, 400,
                                   weights_initializer=_random_uniform_initializer(self.states),
                                   biases_initializer=_random_uniform_initializer(self.states))
        net = slim.fully_connected(net, 300,
                                   weights_initializer=_random_uniform_initializer(net),
                                   biases_initializer=_random_uniform_initializer(net))
        net = slim.fully_connected(net, Config.data.action_dim, activation_fn=tf.nn.tanh,
                                   weights_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003),
                                   biases_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003), )
        self.actions = net * Config.data.action_bound  # Scale actions

    def build_train_op(self, qa_value):
        self.loss = -tf.reduce_mean(qa_value)
        self.train_op = tf.train.AdamOptimizer(Config.train.actor_lr).minimize(self.loss, var_list=self.params)
