import numpy as np
import tensorflow as tf

from hooks import TrainHook, EvalHook
from architecture import ActorGraph, CriticGraph
from replay_memory import ExperienceReplay
from utils import Config


class Agent:
    def __init__(self, env):
        self.replay_memory = ExperienceReplay(Config.train.memory_size)
        self.env = env
        self.sess = None
        self._config_initialize()

    def _config_initialize(self):
        """initialize env config"""
        Config.data.action_dim = self.env.action_space.shape[0]
        Config.data.action_bound = self.env.action_space.high[0]
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
            training_hooks=self.training_hooks,
            evaluation_hooks=self.evaluation_hooks)

    def build_graph(self):
        actions = ActorGraph('eval').build(self.states)
        actions = tf.identity(actions, 'eval/actions')

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_loss(actions)
            self._build_train_op()
        else:
            self.loss = tf.constant(1)
            self.evaluation_hooks = [EvalHook(self.env, self)]

    def _build_loss(self, actions):
        self.ou_noise = OUNoise(Config.data.action_dim, Config.train.noise_theta, Config.train.noise_sigma,
                                Config.train.noise_mu)

        next_state = tf.placeholder(tf.float32, [Config.train.batch_size, Config.data.state_dim], 'next_state')

        q_eval = CriticGraph('eval').build(self.states, actions)
        next_actions = ActorGraph('target').build(next_state)
        q_next = CriticGraph('target').build(next_state, next_actions)
        tf.identity(q_next, 'q_next')
        q_target = tf.placeholder(tf.float32, [Config.train.batch_size, 1], 'q_target')

        self.actor_loss = -tf.reduce_mean(q_eval)
        self.critic_loss = tf.reduce_mean(tf.square(q_target - q_eval)) + tf.losses.get_regularization_loss(
            'critic/eval')
        self.loss = self.actor_loss

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

        self.training_hooks = [TrainHook(self)]

    def choose_action(self, observation, noise=True):
        """choose action from actor eval net"""
        observation = [observation]
        action = self.sess.run('eval/actions:0', feed_dict={self.states: observation})[0]
        if noise:
            noise = self.ou_noise.noise()  # add exploration noise
            action = np.clip(action + noise, -Config.data.action_bound, Config.data.action_bound)
        return action

    def store_transition(self, *args):
        self.replay_memory.add(args)

    def learn(self):
        batch = self.replay_memory.get_batch(Config.train.batch_size)
        states, actions, rewards, next_states = [], [], [], []

        for d in batch:
            states.append(d[0])
            actions.append(d[1])
            rewards.append(d[2])
            next_states.append(d[3])

        q_next = self.sess.run('q_next:0', {'next_state:0': next_states})
        q_target = []

        for i in range(len(batch)):
            done = batch[i][4]
            if done:
                q_target.append([rewards[i]])
            else:
                q_target.append([rewards[i] + Config.train.reward_decay * q_next[i][0]])

        self.sess.run(self.critic_train_op,
                      {'current_state:0': states, 'eval/actions:0': actions, 'q_target:0': q_target})

        return tf.train.SessionRunArgs('global_step:0',
                                       feed_dict={'current_state:0': states})


class OUNoise:
    """Ornstein–Uhlenbeck process"""

    def __init__(self, action_dim, theta=0.15, sigma=0.2, mu=0):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
