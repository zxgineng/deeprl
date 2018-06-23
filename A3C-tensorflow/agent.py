import numpy as np
import tensorflow as tf
import multiprocessing
import copy

from actor import Actor
from critic import Critic
from utils import Config, Scaler
from hooks import TrainingHook, EvalHook


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
        chief = Agent(copy.deepcopy(self.env), 'chief')
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            workers = []
            for i in range(multiprocessing.cpu_count()):
                worker = Agent(copy.deepcopy(self.env), 'worker' + str(i), chief)
                workers.append(worker)

            ave_ep_reward = tf.placeholder(tf.float32, name='ave_ep_reward')
            tf.summary.scalar('ave_ep_reward', ave_ep_reward)
            self.loss = ave_ep_reward
            self.train_op = tf.no_op()
            self.training_hooks = [TrainingHook(chief, workers)]
        else:
            self.loss = tf.constant(1)
            self.evaluation_hooks = [EvalHook(chief)]


class Agent:
    def __init__(self, env, name, chief=None):
        assert name == 'chief' or 'worker' in name
        if 'worker' in name:
            assert chief is not None
            self.chief = chief
        else:
            self.scaler = Scaler(Config.data.state_dim)
        self.name = name
        self.env = env
        self.sess = None
        self.coord = None

        with tf.variable_scope(name):
            self._build_graph()

    def _build_graph(self):
        self.actor = Actor()
        self.critic = Critic()
        if 'worker' in self.name:
            self._build_update_op()

    def _build_update_op(self):
        global_step = tf.train.get_global_step()
        tf.assign_add(global_step, 1, name='global_step_add')

        with tf.variable_scope('sync'):
            with tf.variable_scope('pull'):
                pull_a_params_op = [actor_param.assign(chief_param) for actor_param, chief_param in
                                    zip(self.actor.params, self.chief.actor.params)]
                pull_c_params_op = [critic_param.assign(chief_param) for critic_param, chief_param in
                                    zip(self.critic.params, self.chief.critic.params)]
                self.pull_op = tf.group(pull_a_params_op + pull_c_params_op)

            with tf.variable_scope('push'):
                update_a_op = self.chief.actor.optimizer.apply_gradients(
                    zip(self.actor.grads, self.chief.actor.params))
                update_c_op = self.chief.critic.optimizer.apply_gradients(
                    zip(self.critic.grads, self.chief.critic.params))
                self.update_op = tf.group([update_a_op, update_c_op])

    def init_scaler(self, init_episode=5):
        for e in range(init_episode):
            observation = self.env.reset()
            states = []
            done = False
            count = 0
            while not done:
                states.append(observation)
                action = self.choose_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                observation = next_observation

                if Config.train.get('max_episode_steps', None):
                    count += 1
                    if count == Config.train.max_episode_steps:
                        break
            self.scaler.update(np.array(states))

    def update_chief(self, states, actions, target_v):
        feed_dict = {self.critic.states: states}
        value = self.sess.run(self.critic.value, feed_dict)
        td_error = np.array(target_v) - value
        feed_dict = {self.critic.states: states,
                     self.critic.target_v: target_v,
                     self.actor.states: states,
                     self.actor.actions: actions,
                     self.actor.td_error: td_error}
        self.sess.run([self.critic.loss, self.update_op, self.name + '/global_step_add:0'], feed_dict)

    def pull_params(self):
        self.sess.run(self.pull_op)

    def cal_target_v(self, done, next_observation, rewards):
        if done:
            next_value = 0
        else:
            next_value = self.sess.run(self.critic.value, {self.critic.states: [self.chief.scaler.normalize(next_observation)]})[0, 0]
        target_v = []
        for reward in rewards[::-1]:
            next_value = reward + Config.train.reward_decay * next_value
            target_v.append([next_value])
        target_v.reverse()
        return target_v

    def choose_action(self, observation):
        if Config.data.action_type == 'discrete':
            policy = self.sess.run(self.actor.policy, {self.actor.states: [observation]})[0]
            action = np.random.choice(range(Config.data.action_num), p=policy)
        else:
            action = self.sess.run(self.actor.sample, {self.actor.states: [observation]})  # [action_num]
        return action

    def eval(self, animate=False):
        assert self.name == 'chief'
        observation = self.env.reset()
        ep_reward = 0
        count = 0
        done = False
        while not done:
            if animate:
                self.env.render()
            action = self.choose_action(self.scaler.normalize(observation))
            next_observation, reward, done, info = self.env.step(action)
            ep_reward += reward
            observation = next_observation

            if Config.train.get('max_episode_steps',None):
                count += 1
                if count == Config.train.max_episode_steps:
                    break
        return ep_reward

    def work(self):
        total_step = 0
        states, actions, rewards, unscaled_states = [], [], [], []
        self.pull_params()

        while not self.coord.should_stop():
            observation = self.env.reset()
            ep_reward = 0
            done = False
            count = 0
            while not done:
                unscaled_states.append(observation)
                observation = self.chief.scaler.normalize(observation)
                states.append(observation)
                action = self.choose_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                total_step += 1
                ep_reward += reward


                actions.append(action)
                rewards.append(reward)
                if total_step % Config.train.update_n_iter == 0 or done:
                    target_v = self.cal_target_v(done, next_observation, rewards)
                    self.update_chief(states, actions, target_v)
                    self.chief.scaler.update(np.array(unscaled_states))
                    states, actions, rewards, unscaled_states = [], [], [], []
                    self.pull_params()

                observation = next_observation

                if Config.train.get('max_episode_steps', None):
                    count += 1
                    if count == Config.train.max_episode_steps:
                        break


