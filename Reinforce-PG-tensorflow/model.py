import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config


class Model:
    def __init__(self):
        with tf.variable_scope('model') as sc:
            self._build_graph()
            self.params = tf.trainable_variables(sc.name)
            self._build_loss()
            self._build_train_op()

    def _build_graph(self):
        with tf.variable_scope('network'):
            self.states = tf.placeholder(tf.float32, [None, Config.data.state_dim], 'states')
            net = slim.fully_connected(self.states, Config.data.state_dim * 10)
            self.logits = slim.fully_connected(net, Config.data.action_num, None)
            self.prob = tf.nn.softmax(self.logits)

    def _build_loss(self):
        with tf.variable_scope('loss'):
            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            self.value = tf.placeholder(tf.float32, [None], 'value')
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits) * self.value)

    def _build_train_op(self):
        with tf.variable_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(Config.train.learning_rate).minimize(self.loss,var_list=self.params)
