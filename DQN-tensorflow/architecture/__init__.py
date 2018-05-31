import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config


class Graph:
    def __init__(self,name):
        self.name = name

    def build(self, state,num_action):
        with tf.variable_scope(self.name):
            with tf.variable_scope('fc1'):
                net = slim.fully_connected(state,Config.model.fc1_unit)
            with tf.variable_scope('fc2'):
                outputs = slim.fully_connected(net,num_action,activation_fn=None)
            return outputs

