import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config


class ActorGraph:
    def __init__(self, name):
        self.name = name

    def build(self, inputs):
        with tf.variable_scope('actor/' + self.name):
            net = slim.fully_connected(inputs, 30)
            net = slim.fully_connected(net, Config.data.action_dim, activation_fn=tf.nn.tanh)
            outputs = net * 2  # Scale outputs to [-2 ~ 2]
            return outputs


class CriticGraph:
    def __init__(self, name):
        self.name = name

    def build(self, inputs, actions):
        with tf.variable_scope('critic/' + self.name):
            net1 = slim.fully_connected(inputs, 30, activation_fn=None)
            net2 = slim.fully_connected(actions, 30, activation_fn=None, biases_initializer=None)
            net = tf.nn.relu(net1 + net2)
            outputs = slim.fully_connected(net, Config.data.action_dim, activation_fn=None)
            return outputs
