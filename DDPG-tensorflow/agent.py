import numpy as np
import tensorflow as tf
import gym

from train_hooks import AlgoTrainHook
from architecture import ActorGraph, CriticGraph
from replay_memory import ExperienceReplay
from utils import Config


class Agent:
    def __init__(self):
        self.replay_memory = ExperienceReplay(Config.train.memory_size)
        self.sess = None

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.states = features
        self.loss, self.train_op, self.predictions, self.training_hooks = None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.actor_loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks)

    def build_graph(self):
        actions = ActorGraph('eval').build(self.states)
        actions = tf.identity(actions, 'eval/actions')

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(actions)
            self._build_train_op()

    def _build_loss(self, actions):
        self.noise = 3
        next_state = tf.placeholder(tf.float32, [Config.train.batch_size, Config.data.state_dim], 'next_state')
        rewards = tf.placeholder(tf.float32, [Config.train.batch_size, 1], 'rewards')

        q_eval = CriticGraph('eval').build(self.states, actions)
        next_actions = ActorGraph('target').build(next_state)
        q_next = CriticGraph('target').build(next_state, next_actions)
        q_target = rewards + Config.train.reward_decay * q_next

        self.actor_loss = - tf.reduce_mean(q_eval)
        self.critic_loss = tf.reduce_mean(tf.square(q_target - q_eval))

    def _build_train_op(self):
        global_step = tf.train.get_or_create_global_step()
        self.actor_train_op = tf.train.AdamOptimizer(Config.train.actor_learning_rate).minimize(self.actor_loss,
                                                                                                global_step,
                                                                                                tf.trainable_variables(
                                                                                                    'actor/eval'))
        self.critic_train_op = tf.train.AdamOptimizer(Config.train.critic_learning_rate).minimize(self.critic_loss,
                                                                                                  var_list=tf.trainable_variables(
                                                                                                      'critic/eval'))
        self.train_op = self.actor_train_op

        env = gym.make('Pendulum-v0').unwrapped
        self.training_hooks = [AlgoTrainHook(env, self)]

    def choose_action(self, observation):
        """choose action from actor eval net"""
        observation = np.expand_dims(observation, 0)
        action = self.sess.run('eval/actions:0', feed_dict={self.states: observation})
        exploration_noise = np.random.normal(0, self.noise)  # add exploration noise
        action = np.clip(action + exploration_noise, -2, 2)
        return action[0]

    def store_transition(self, *args):
        self.replay_memory.add(args)

    def learn(self):
        self.noise = self.noise * 0.9995
        batch = self.replay_memory.get_batch(Config.train.batch_size)
        states, actions, rewards, next_states = [], [], [], []
        for d in batch:
            states.append(d[0])
            actions.append(d[1])
            rewards.append([d[2]])  # rewards shape [N,1]
            next_states.append(d[3])

        self.sess.run(self.critic_train_op, {'current_state:0': states, 'eval/actions:0': actions, 'rewards:0': rewards,
                                             'next_state:0': next_states})

        return tf.train.SessionRunArgs('global_step:0',
                                       feed_dict={'current_state:0': states})
