import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config


class Graph:
    def __init__(self):
        pass

    def build(self, inputs):

        with tf.variable_scope('fc1'):
            net = slim.fully_connected(inputs, 10)

        with tf.variable_scope('fc2'):
            logits = slim.fully_connected(net, Config.data.action_num, activation_fn=None)
            return logits
