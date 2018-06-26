import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from utils import Config


def _random_uniform_initializer(x):
    num = x.shape.as_list()[-1]
    return tf.random_uniform_initializer(
        minval=-1.0 / np.sqrt(num), maxval=1.0 / np.sqrt(num))


class Critic:
    def __init__(self, actions):
        with tf.variable_scope('critic') as sc:
            self.actions = actions
            self._build_graph()
            self.params = tf.trainable_variables(sc.name)
            self._build_loss()
            self._build_train_op()

    def _build_graph(self):
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(Config.train.critic_l2_loss_weight)):
            self.states = tf.placeholder(tf.float32, [None, Config.data.state_dim], 'states')
            net = slim.fully_connected(self.states, 400, weights_initializer=_random_uniform_initializer(self.states),
                                       biases_initializer=_random_uniform_initializer(self.states))
            net1 = slim.fully_connected(net, 300, activation_fn=None,
                                        weights_initializer=_random_uniform_initializer(net),
                                        biases_initializer=_random_uniform_initializer(net))
            net2 = slim.fully_connected(self.actions, 300, activation_fn=None,
                                        weights_initializer=_random_uniform_initializer(self.actions),
                                        biases_initializer=None)
            net = tf.nn.relu(net1 + net2)
            self.qa_value = slim.fully_connected(net, 1, activation_fn=None,
                                                 weights_initializer=tf.random_uniform_initializer(minval=-0.0003,
                                                                                                   maxval=0.0003),
                                                 biases_initializer=tf.random_uniform_initializer(minval=-0.0003,
                                                                                                  maxval=0.0003))

    def _build_loss(self):
        self.target_qa = tf.placeholder(tf.float32, [Config.train.batch_size, 1], 'target_qa')
        sc = tf.get_default_graph().get_name_scope()
        self.loss = tf.reduce_mean(tf.square(self.target_qa - self.qa_value)) + tf.losses.get_regularization_loss(sc)


    def _build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(Config.train.critic_lr).minimize(self.loss, var_list=self.params)
