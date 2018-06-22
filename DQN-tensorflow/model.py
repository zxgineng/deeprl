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
            net = slim.fully_connected(self.states, Config.data.state_dim*10)
            self.q_value = slim.fully_connected(net, Config.data.action_num, activation_fn=None)

    def _build_loss(self):
        with tf.variable_scope('loss'):
            self.y = tf.placeholder(tf.float32, [Config.train.batch_size], 'target_q')
            self.actions = tf.placeholder(tf.int32, [Config.train.batch_size], 'actions')
            actions = tf.one_hot(self.actions, Config.data.action_num)
            qa_value = tf.reduce_sum(self.q_value * actions, -1)
            td_error = tf.abs(self.y - qa_value)
            self.loss = tf.reduce_mean(tf.square(td_error))

    def _build_train_op(self):
        with tf.variable_scope('optimizer'):
            self.train_op = tf.train.RMSPropOptimizer(Config.train.learning_rate).minimize(self.loss,
                                                                                           var_list=self.params)
