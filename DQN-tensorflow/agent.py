import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from replay_memory import ExperienceReplay
from utils import Config
from architecture import Graph
from hooks import TrainHook


class Agent:

    def __init__(self, env):
        self.replay_memory = ExperienceReplay(Config.train.memory_size)
        self.env = env
        self.sess = None
        self._config_initialize()

    def _config_initialize(self):
        """initialize env config"""
        if 'n' not in vars(self.env.action_space):
            Config.data.action_dim = self.env.action_space.shape[0]
            Config.data.action_bound = self.env.action_space.high[0]
            Config.data.action_type = 'continuous'
        else:
            Config.data.action_num = self.env.action_space.n
            Config.data.action_type = 'discrete'
        Config.data.state_dim = self.env.observation_space.shape[0]

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.states = features
        self.loss, self.train_op, self.predictions, self.training_hooks, self.evaluation_hooks = None, None, None, None, None
        self.build_graph()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks)

    def build_graph(self):
        graph = Graph('eval_net')

        q_eval = graph.build(self.states)
        tf.identity(q_eval,'q_eval')

        self.predictions = q_eval

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_loss(q_eval)
            self._build_train_op()

    def _build_loss(self,q_eval):
        graph = Graph('target_net')
        next_state = tf.placeholder(tf.float32,
                                    [Config.train.batch_size, Config.data.state_dim], 'next_state')
        q_next = graph.build(next_state)
        tf.identity(q_next,'q_next')

        y = tf.placeholder(tf.float32, [None], 'q_target')
        action = tf.placeholder(tf.float32, [None, Config.data.action_num], 'action')
        q_a_eval = tf.reduce_sum(q_eval * action, -1)
        self.loss = tf.reduce_mean(tf.squared_difference(y, q_a_eval))

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, self.global_step,
            optimizer=tf.train.RMSPropOptimizer(Config.train.learning_rate),
            learning_rate=Config.train.learning_rate,
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'eval_net'),
            name="train_op")
        self.training_hooks = [TrainHook(self)]

    def choose_action(self, observation, epsilon, one_hot=False):

        observation = np.expand_dims(observation, 0)

        if np.random.uniform() >= epsilon:
            actions_value = self.sess.run('q_eval:0', feed_dict={'current_state:0': observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, Config.data.action_num)

        if one_hot:
            action_index = action
            action = np.zeros(Config.data.action_num)
            action[action_index] = 1
        return action

    def store_transition(self,*args):
        self.replay_memory.add(args)

    def learn(self):
        batch = self.replay_memory.get_batch(Config.train.batch_size)

        state, action, reward, next_state = [], [], [], []
        for d in batch:
            state.append(d[0])
            if not isinstance(d[1],(int,np.int64)):
                action.append(d[1])
            else:
                one_hot = [0] * Config.data.action_num
                one_hot[d[1]] = 1
                action.append(one_hot)
            reward.append(d[2])
            next_state.append(d[3])

        y = []
        q_next = self.sess.run('q_next:0', feed_dict={'next_state:0': next_state})

        for i in range(0, len(batch)):
            terminal = batch[i][4]
            # if terminal, only equals reward
            if terminal:
                y.append(reward[i])
            else:
                y.append(reward[i] + Config.train.reward_decay * np.max(q_next[i]))

        return tf.train.SessionRunArgs('global_step:0',
                                       feed_dict={'q_target:0': y, 'current_state:0': state, 'action:0': action})
