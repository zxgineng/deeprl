import numpy as np
import tensorflow as tf

from utils import Config
from model import Model
from hooks import TrainingHook, EvalHook


class Agent:
    def __init__(self, env):
        self.env = env
        self.sess = None
        self._config_initialize()

    def _config_initialize(self):
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
        self.model = Model()

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            ave_ep_reward = tf.placeholder(tf.float32, name='ave_ep_reward')
            tf.summary.scalar('ave_reward', ave_ep_reward)
            self.loss = ave_ep_reward
            global_step = tf.train.get_global_step()
            self.train_op = tf.assign_add(global_step, 1)
            self.training_hooks = [TrainingHook(self)]
        else:
            self.loss = tf.constant(0)
            self.evaluation_hooks = [EvalHook(self)]

    def train(self, states, actions, rewards):
        value = []
        next_value = 0
        for reward in rewards[::-1]:
            next_value = reward + Config.train.reward_decay * next_value
            value.append(next_value)
        value.reverse()

        self.sess.run(self.model.train_op,
                      {self.model.states: states, self.model.actions: actions, self.model.value: value})

    def eval(self, animate=False):
        observation = self.env.reset()
        done = False
        ep_reward = 0
        while not done:
            if animate:
                self.env.render()
            action = self.choose_action(observation)
            next_observation, reward, done, info = self.env.step(action)
            ep_reward += reward
            observation = next_observation

        return ep_reward

    def run_episode(self, animate=False):
        observation = self.env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            if animate:
                self.env.render()
            states.append(observation)
            action = self.choose_action(observation)
            actions.append(action)
            next_observation, reward, done, info = self.env.step(action)

            # customize reward
            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = next_observation
            r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
            r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
            reward = r1 + r2

            rewards.append(reward)
            observation = next_observation

        return states, actions, rewards

    def choose_action(self, observation):
        prob = self.sess.run(self.model.prob, {self.model.states: [observation]})
        action = np.random.choice(range(prob.shape[1]), p=prob.ravel())
        return action

