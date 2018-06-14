import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config


class ActorGraph:
    def __init__(self):
        pass

    def build(self, inputs):
        with tf.variable_scope('actor'):
            net = slim.fully_connected(inputs, 200)
            if Config.data.action_type == 'discrete':
                outputs = slim.fully_connected(net, Config.data.action_num, activation_fn=None)
                return outputs
            else:
                mu = slim.fully_connected(net, Config.data.action_dim, tf.nn.tanh)
                sigma = slim.fully_connected(net, Config.data.action_dim, tf.nn.softplus)
                return mu, sigma

class CriticGraph:
    def __init__(self):
        pass

    def build(self, inputs):
        with tf.variable_scope('critic'):
            net = slim.fully_connected(inputs, 200)
            value = slim.fully_connected(net, 1, activation_fn=None)
            return value
