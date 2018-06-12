import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from utils import Config


def _random_uniform_initializer(num):
    return tf.random_uniform_initializer(
        minval=-1.0 / np.sqrt(num), maxval=1.0 / np.sqrt(num))


class ActorGraph:
    def __init__(self, name):
        self.name = name

    def build(self, inputs):
        with tf.variable_scope('actor/' + self.name):
            net = slim.fully_connected(inputs, 400,
                                       weights_initializer=_random_uniform_initializer(inputs.shape.as_list()[-1]),
                                       biases_initializer=_random_uniform_initializer(inputs.shape.as_list()[-1]))
            net = slim.fully_connected(net, 300,
                                       weights_initializer=_random_uniform_initializer(net.shape.as_list()[-1]),
                                       biases_initializer=_random_uniform_initializer(net.shape.as_list()[-1]))
            net = slim.fully_connected(net, Config.data.action_dim, activation_fn=tf.nn.tanh,
                                       weights_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003),
                                       biases_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003), )
            outputs = net * Config.data.action_bound  # Scale outputs
            return outputs


class CriticGraph:
    def __init__(self, name):
        self.name = name

    def build(self, inputs, actions):
        with tf.variable_scope('critic/' + self.name):
            with slim.arg_scope([slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(Config.train.critic_l2_loss_weight)):
                net = slim.fully_connected(inputs, 400,
                                           weights_initializer=_random_uniform_initializer(inputs.shape.as_list()[-1]),
                                           biases_initializer=_random_uniform_initializer(inputs.shape.as_list()[-1]))
                net1 = slim.fully_connected(net, 300, activation_fn=None,
                                            weights_initializer=_random_uniform_initializer(net.shape.as_list()[-1]),
                                            biases_initializer=_random_uniform_initializer(net.shape.as_list()[-1]))
                net2 = slim.fully_connected(actions, 300, activation_fn=None,
                                            weights_initializer=_random_uniform_initializer(net.shape.as_list()[-1]),
                                            biases_initializer=None)
                net = tf.nn.relu(net1 + net2)
                outputs = slim.fully_connected(net, 1, activation_fn=None,
                                               weights_initializer=tf.random_uniform_initializer(minval=-0.0003,
                                                                                                 maxval=0.0003),
                                               biases_initializer=tf.random_uniform_initializer(minval=-0.0003,
                                                                                                maxval=0.0003), )
                return outputs


