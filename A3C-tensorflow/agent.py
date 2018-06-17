import numpy as np
import tensorflow as tf
import multiprocessing
import copy

from architecture import Graph
from utils import Config
from hooks import TrainingHook


class Model:
    def __init__(self, env):
        self.env = env
        self.sess = None
        self._config_initialize()

    def _config_initialize(self):
        """initialize env config"""
        if 'n' not in vars(self.env.action_space):
            Config.data.action_dim = self.env.action_space.shape[0]
            Config.data.action_bound = self.env.action_space.high
            Config.data.action_type = 'continuous'
        else:
            Config.data.action_num = self.env.action_space.n
            Config.data.action_type = 'discrete'
        Config.data.state_dim = self.env.observation_space.shape[0]

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.loss, self.train_op, self.predictions, self.training_hooks = None, None, None, None
        self.build_graph()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks)

    def build_graph(self):
        with tf.device('/cpu:0'):
            chief = Agent(copy.deepcopy(self.env), 'chief')
            chief.build_graph()
            workers = []
            for i in range(multiprocessing.cpu_count()):
                worker = Agent(copy.deepcopy(self.env), 'worker' + str(i), chief)
                worker.build_graph()
                workers.append(worker)

            if self.mode == tf.estimator.ModeKeys.TRAIN:
                ave_ep_reward = tf.placeholder(tf.float32, name='ave_ep_reward')
                tf.summary.scalar('ave_ep_reward', ave_ep_reward)
                self.loss = ave_ep_reward
                self.train_op = tf.no_op()
                self.training_hooks = [TrainingHook(chief, workers)]


class Agent:
    def __init__(self, env, mode, chief=None):
        assert mode == 'chief' or 'worker' in mode
        if 'worker' in mode:
            assert chief is not None
            self.chief = chief
        self.mode = mode
        self.env = env
        self.sess = None
        self.coord = None

    def build_graph(self):
        if self.mode == 'chief':
            with tf.variable_scope(self.mode):
                self.states = tf.placeholder(tf.float32, [None, Config.data.state_dim])

                if Config.data.action_type == 'discrete':
                    _, logits = Graph().build(self.states)
                    tf.nn.softmax(logits, name='action_prob')
                else:
                    _, mu, sigma = Graph().build(self.states)
                    mu, sigma = Config.data.action_bound * mu, sigma
                    normal_dist = tf.distributions.Normal(mu, sigma)
                    tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), -Config.data.action_bound,
                                     Config.data.action_bound, 'continuous_action')

                self.actor_params = tf.trainable_variables('chief/actor')
                self.critic_params = tf.trainable_variables('chief/critic')

                self.actor_optimizer = tf.train.RMSPropOptimizer(Config.train.actor_learning_rate, decay=0.99)
                self.critic_optimizer = tf.train.RMSPropOptimizer(Config.train.critic_learning_rate, decay=0.99)

        else:
            with tf.variable_scope(self.mode):
                self.states = tf.placeholder(tf.float32, [None, Config.data.state_dim], 'states')
                if Config.data.action_type == 'discrete':
                    self.actions = tf.placeholder(tf.int32, [None], 'actions')
                else:
                    self.actions = tf.placeholder(tf.float32, [None, Config.data.action_dim], 'actions')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'v_target')

                if Config.data.action_type == 'discrete':
                    self.value, logits = Graph().build(self.states)
                    action_prob = tf.nn.softmax(logits, name='action_prob')
                    self._build_discrete_loss(action_prob)
                else:
                    self.value, mu, sigma = Graph().build(self.states)
                    mu, sigma = Config.data.action_bound * mu, sigma

                    normal_dist = tf.distributions.Normal(mu, sigma)
                    tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), -Config.data.action_bound,
                                     Config.data.action_bound, 'continuous_action')
                    self._build_continuous_loss(normal_dist)

                self._build_update_op()

    def _build_discrete_loss(self, action_prob):
        td_error = self.v_target - self.value

        with tf.variable_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(td_error))

        with tf.variable_scope('actor_loss'):
            log_prob = tf.reduce_sum(
                tf.log(action_prob) * tf.one_hot(self.actions, Config.data.action_num, dtype=tf.float32), axis=1,
                keep_dims=True)
            exp_v = log_prob * tf.stop_gradient(td_error)
            entropy = -tf.reduce_sum(action_prob * tf.log(action_prob + 1e-5),
                                     axis=1, keep_dims=True)
            self.actor_loss = tf.reduce_mean(-(Config.train.dis_entropy_beta * entropy + exp_v))

    def _build_continuous_loss(self, normal_dist):
        td_error = self.v_target - self.value

        with tf.variable_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(td_error))
        with tf.variable_scope('actor_loss'):
            log_prob = normal_dist.log_prob(self.actions)
            entropy = normal_dist.entropy()
            exp_v = log_prob * tf.stop_gradient(td_error)
            self.actor_loss = tf.reduce_mean(-(Config.train.con_entropy_beta * entropy + exp_v))

    def _build_update_op(self):
        actor_params = tf.trainable_variables(self.mode + '/actor')
        critic_params = tf.trainable_variables(self.mode + '/critic')
        global_step = tf.train.get_global_step()
        tf.assign_add(global_step, 1, name='global_step_add')

        with tf.variable_scope('worker_grads'):
            actor_grads = tf.gradients(self.actor_loss, actor_params)
            critic_grads = tf.gradients(self.critic_loss, critic_params)

        with tf.variable_scope('sync'):
            with tf.variable_scope('pull'):
                pull_actor_params = [tf.assign(worker_param, chief_param) for worker_param, chief_param in
                                     zip(actor_params, self.chief.actor_params)]
                pull_critic_params = [tf.assign(worker_param, chief_param) for worker_param, chief_param in
                                      zip(critic_params, self.chief.critic_params)]
                self.pull_op = tf.group(pull_actor_params + pull_critic_params)

            with tf.variable_scope('push'):
                push_actor_params = self.chief.actor_optimizer.apply_gradients(
                    zip(actor_grads, self.chief.actor_params))
                push_critic_params = self.chief.critic_optimizer.apply_gradients(
                    zip(critic_grads, self.chief.critic_params))
                self.push_op = tf.group([push_actor_params, push_critic_params])

    def choose_action(self, observation):
        if Config.data.action_type == 'discrete':
            prob = self.sess.run(self.mode + '/action_prob:0', {self.states: [observation]})[0]
            action = np.random.choice(range(Config.data.action_num), p=prob)  # scalar
        else:
            action = self.sess.run(self.mode + '/continuous_action:0', {self.states: [observation]})  # [action_num]
        return action

    def eval(self):
        assert self.mode == 'chief'
        observation = self.env.reset()
        ep_reward = 0
        while True:
            action = self.choose_action(observation)
            next_observation, reward, done, info = self.env.step(action)
            ep_reward += reward
            if done:
                return ep_reward
            else:
                observation = next_observation

    def learn(self):
        assert 'worker' in self.mode
        total_step = 0
        self.sess.run(self.pull_op)
        buffer_states, buffer_actions, buffer_rewards = [], [], []
        try:
            while not self.coord.should_stop():
                observation = self.env.reset()
                ep_reward = 0
                while True:
                    action = self.choose_action(observation)
                    next_observation, reward, done, info = self.env.step(action)
                    total_step += 1
                    ep_reward += reward
                    buffer_states.append(observation)
                    buffer_actions.append(action)
                    buffer_rewards.append(reward)

                    if total_step % Config.train.update_n_iter == 0 or done:

                        if done:
                            next_value = 0
                        else:
                            next_value = self.sess.run(self.value, {self.states: [next_observation]})[0, 0]
                        buffer_v_target = []

                        for reward in buffer_rewards[::-1]:
                            next_value = reward + Config.train.reward_decay * next_value
                            buffer_v_target.append([next_value])
                        buffer_v_target.reverse()

                        self.sess.run([self.push_op, self.mode + '/global_step_add:0'],
                                      {self.states: buffer_states, self.actions: buffer_actions,
                                       self.v_target: buffer_v_target})
                        buffer_states, buffer_actions, buffer_rewards = [], [], []
                        self.sess.run(self.pull_op)

                    if done:
                        if Config.train.global_ep == 0:
                            Config.train.global_running_r = ep_reward
                        else:
                            Config.train.global_running_r = 0.9 * Config.train.global_running_r + 0.1 * ep_reward
                        Config.train.global_ep += 1
                        if Config.train.global_ep % 100 == 0:
                            print(self.mode, '  global_ep:', Config.train.global_ep,
                                  '    global_running_r:',
                                  round(Config.train.global_running_r, 2))
                        break
                    else:
                        observation = next_observation
        except Exception as e:
            pass
