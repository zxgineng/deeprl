import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config


class Graph:
    def __init__(self):
        pass

    def build(self,inputs):
        with tf.variable_scope('critic'):
            net = slim.fully_connected(inputs, 200)
            c_net = slim.fully_connected(net,100)
            value = slim.fully_connected(c_net, 1, activation_fn=None)

        with tf.variable_scope('actor'):
            a_net = slim.fully_connected(net, 100)
            if Config.data.action_type == 'discrete':
                outputs = slim.fully_connected(a_net, Config.data.action_num, activation_fn=None)
                return value,outputs
            else:
                # mu = slim.fully_connected(net, Config.data.action_dim, None)
                # log_stdd = tf.get_variable('log_stdd', [1, Config.data.action_dim], tf.float32,
                #                            tf.constant_initializer(0.0))
                # sigma = tf.exp(log_stdd)
                mu = slim.fully_connected(a_net, Config.data.action_dim)
                sigma = slim.fully_connected(a_net, Config.data.action_dim, tf.nn.softplus)
                return value,mu, sigma
