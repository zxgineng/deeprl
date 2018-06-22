import numpy as np
import tensorflow as tf

from hooks import TrainingHook, EvalHook
from model import Model
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
        Config.data.action_num = self.env.action_space.n
        Config.data.state_dim = self.env.observation_space.shape[0]

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.loss, self.train_op, self.predictions, self.training_hooks, self.evaluation_hooks = None, None, None, None, None
        self._build_graph()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks,
            evaluation_hooks=self.evaluation_hooks)

    def _build_graph(self):
        with tf.variable_scope('eval_net'):
            self.eval_net = Model()

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_update_op()
            ave_ep_reward = tf.placeholder(tf.float32, None, name='ave_ep_reward')
            tf.summary.scalar('ave_ep_reward', ave_ep_reward)
            self.loss = ave_ep_reward
            self.train_op = tf.no_op()
            self.training_hooks = [TrainingHook(self)]
        else:
            self.loss = tf.constant(0)
            self.evaluation_hooks = [EvalHook(self)]

    def _build_update_op(self):
        with tf.variable_scope('target_net'):
            self.target_net = Model()
        global_step = tf.train.get_global_step()
        tf.assign_add(global_step, 1, name='global_step_add')
        self.replace_target_op = tf.group([tf.assign(target_param, train_param) for train_param, target_param in
                                           zip(self.eval_net.params, self.target_net.params)])

    def eval(self, animate=False):
        observation = self.env.reset()
        done = False
        ep_reward = 0
        count = 0
        while not done:
            if animate:
                self.env.render()
            action = self.choose_action(observation)
            next_observation, reward, done, info = self.env.step(action)
            ep_reward += reward
            observation = next_observation

            if Config.train.get('max_episode_steps',None):
                count += 1
                if count == Config.train.max_episode_steps:
                    break
        return ep_reward

    def run_episode(self, animate=False):
        observation = self.env.reset()
        count = 0
        ep_reward = 0
        done = False
        while not done:
            if animate:
                self.env.render()
            action = self.choose_action(observation)
            next_observation, reward, done, info = self.env.step(action)

            # customize reward
            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = next_observation
            r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
            r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
            reward = r1 + r2

            ep_reward += reward
            self.store_transition(observation, action, reward, next_observation, done)
            observation = next_observation

            if self.replay_memory.length >= Config.train.observe_n_iter:
                global_step = self.update_params()
                Config.train.epsilon = max(Config.train.get('final_epsilon',0.0),
                                           Config.train.epsilon - Config.train.epsilon_decrement)

                if global_step % Config.train.replace_target_n_iter == 0:
                    self.replace_target_params()

            if Config.train.get('max_episode_steps',None):
                count += 1
                if count == Config.train.max_episode_steps:
                    break
        return ep_reward

    def choose_action(self, observation):

        if np.random.uniform() >= Config.train.epsilon:
            actions_value = self.sess.run(self.eval_net.q_value, feed_dict={self.eval_net.states: [observation]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, Config.data.action_num)
        return action

    def store_transition(self, *args):
        self.replay_memory.add(args)

    def replace_target_params(self):
        self.sess.run(self.replace_target_op)

    def update_params(self):
        batch = self.replay_memory.get_batch(Config.train.batch_size)

        states, actions, rewards, next_states = [], [], [], []
        for d in batch:
            states.append(d[0])
            actions.append(d[1])
            rewards.append(d[2])
            next_states.append(d[3])

        y = []
        next_q = self.sess.run(self.target_net.q_value, {self.target_net.states: next_states})

        for i in range(len(batch)):
            terminal = batch[i][4]
            # if terminal, only equals reward
            if terminal:
                y.append(rewards[i])
            else:
                y.append(rewards[i] + Config.train.reward_decay * np.max(next_q[i]))

        feed_dict = {self.eval_net.y: y, self.eval_net.states: states, self.eval_net.actions: actions}
        step, _, _ = self.sess.run(
            ['global_step:0', self.eval_net.train_op, 'global_step_add:0'], feed_dict)
        return step
