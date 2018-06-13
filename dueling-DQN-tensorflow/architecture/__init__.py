import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config


class Graph:
    def __init__(self, name):
        self.name = name

    def build(self, inputs):
        with tf.variable_scope(self.name):
            value = slim.fully_connected(inputs, 20)
            value = slim.fully_connected(value, 1, activation_fn=None)
            advantage = slim.fully_connected(inputs, 20)
            advantage = slim.fully_connected(advantage, Config.data.num_action, activation_fn=None)
            logits = value + (advantage - tf.reduce_mean(advantage, -1, True))
            return logits