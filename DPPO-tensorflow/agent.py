import numpy as np
import tensorflow as tf

from hooks import TrainHook
from architecture import ActorGraph, CriticGraph
from utils import Config


class Agent:
    def __init__(self, env):
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
        self.loss, self.train_op, self.predictions, self.training_hooks = None, None, None, None
        self.build_graph()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.actor_loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks)

    def build_graph(self):

        # build placeholder
        if Config.data.action_type == 'discrete':
            self.actions = tf.placeholder(tf.int32, [None], 'actions')
        else:
            self.actions = tf.placeholder(tf.float32, [None, Config.data.action_dim], 'actions')
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'v_target')

        # build network
        self.value = CriticGraph().build(self.states)
        tf.identity(self.value, 'value')
        if Config.data.action_type == 'discrete':
            # todo 离散
            pass
        else:
            with tf.variable_scope('new'):
                mu, sigma = ActorGraph().build(self.states)
                mu, sigma = Config.data.action_bound * mu, sigma + 1e-4
                new_policy = tf.distributions.Normal(mu, sigma)

                tf.clip_by_value(tf.squeeze(new_policy.sample(1), axis=0), -Config.data.action_bound,
                                 Config.data.action_bound, 'continuous_action')

            with tf.variable_scope('old'):
                mu, sigma = ActorGraph().build(self.states)
                mu, sigma = Config.data.action_bound * mu, sigma + 1e-4
                old_policy = tf.distributions.Normal(mu, sigma)

            self._build_continuous_loss(new_policy, old_policy)

        ave_ep_reward = tf.placeholder(tf.float32, name='ave_ep_reward')
        tf.summary.scalar('ave_ep_reward', ave_ep_reward)
        self.loss = ave_ep_reward
        self._build_train_op()

    def _build_discrete_loss(self, action_prob):
        # todo
        pass


    def _build_continuous_loss(self, new_policy, old_policy):
        advantage = self.v_target - self.value
        with tf.variable_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(advantage))
        with tf.variable_scope('actor_loss'):
            ratio = new_policy.prob(self.actions) / old_policy.prob(self.actions)
            surrogate = ratio * advantage
            if Config.train.actor_loss_method == 'kl_pen':
                kl_pen = tf.distributions.kl_divergence(old_policy, new_policy)
                self.kl_mean = tf.reduce_mean(kl_pen)
                self.actor_loss = -tf.reduce_mean(surrogate - Config.train.kl_lam * kl_pen)
            else:  # clip surrogate
                self.actor_loss = -tf.reduce_mean(
                    tf.minimum(surrogate, tf.clip_by_value(ratio, 1 - Config.train.surr_clip_epsilon,
                                                           1 + Config.train.surr_clip_epsilon) * advantage))

    def _build_train_op(self):
        new_actor_params = tf.trainable_variables('new/actor')
        old_actor_params = tf.trainable_variables('old/actor')

        self.critic_train_op = tf.train.AdamOptimizer(Config.train.critic_lr).minimize(self.critic_loss,
                                                                                       var_list=tf.trainable_variables(
                                                                                           'critic'))
        self.actor_train_op = tf.train.AdamOptimizer(Config.train.actor_lr).minimize(self.actor_loss,
                                                                                     var_list=new_actor_params)
        self.update_actor_op = [tf.assign(old_p, new_p) for new_p, old_p in zip(new_actor_params, old_actor_params)]

        self.train_op = tf.no_op()

        self.training_hooks = [TrainHook(self)]

    def choose_action(self, observation):
        if Config.data.action_type == 'discrete':
            prob = self.sess.run('/action_prob:0', {self.states: [observation]})[0]
            action = np.random.choice(range(Config.data.action_num), p=prob)  # scalar
        else:
            action = self.sess.run('/continuous_action:0', {self.states: [observation]})  # [action_num]
        return action

    def learn(self, states, actions, gain):
        self.sess.run(self.update_actor_op)

        if Config.train.actor_loss_method == 'kl_pen':
            for _ in range(Config.train.actor_update_steps):
                _, kl = self.sess.run([self.actor_train_op, self.kl_mean], {self.states: states, self.v_target: gain,
                                                                            self.actions: actions})
                if kl > 4 * Config.train.kl_target:
                    break
            if kl < Config.train.kl_target / Config.train.kl_target_beta:
                Config.train.kl_lam /= Config.train.kl_lam_alpha
            elif kl > Config.train.kl_target * Config.train.kl_target_beta:
                Config.train.kl_lam *= Config.train.kl_lam_alpha
        else:
            [self.sess.run(self.actor_train_op, {self.states: states, self.actions: actions, self.v_target: gain}) for _
             in range(Config.train.actor_update_steps)]
