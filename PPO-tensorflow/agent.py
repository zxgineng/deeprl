import numpy as np
import tensorflow as tf

from hooks import TrainingHook, EvalHook
from actor import Actor
from critic import Critic
from utils import Config, Scaler


class Agent:
    def __init__(self, env):
        self.env = env
        self.sess = None
        self._config_initialize()

    def _config_initialize(self):
        """initialize env config"""
        Config.data.action_dim = self.env.action_space.shape[0]
        Config.data.action_bound = self.env.action_space.high
        Config.data.state_dim = self.env.observation_space.shape[0]

        self.scaler = Scaler(Config.data.state_dim)

    def model_fn(self, mode, features, labels):
        self.mode = mode
        self.loss, self.train_op, self.training_hooks, self.evaluation_hooks = None, None, None, None
        self._build_graph()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            training_hooks=self.training_hooks,
            evaluation_hooks=self.evaluation_hooks)

    def _build_graph(self):
        self.actor = Actor()
        self.critic = Critic()

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            ave_ep_reward = tf.placeholder(tf.float32, name='ave_ep_reward')
            tf.summary.scalar('ave_ep_reward', ave_ep_reward)
            self.loss = ave_ep_reward
            global_step = tf.train.get_global_step()
            self.train_op = tf.assign_add(global_step, 1)
            self.training_hooks = [TrainingHook(self)]
        else:
            self.loss = tf.constant(1)
            self.evaluation_hooks = [EvalHook(self)]

    def init_scaler(self, init_episode=5):
        for e in range(init_episode):
            self.run_episode()

    def choose_action(self, observation):
        feed_dict = {self.actor.states: [observation]}
        action = self.sess.run(self.actor.sample, feed_dict)
        return action

    def eval(self, animate=False):
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

            if Config.train.get('max_episode_steps', None):
                count += 1
                if count == Config.train.max_episode_steps:
                    break

        return ep_reward

    def run_episode(self, animate=False):
        observation = self.env.reset()
        states, actions, rewards, unscaled_states = [], [], [], []
        done = False
        count = 0

        while not done:
            if animate:
                self.env.render()
            unscaled_states.append(observation)
            observation = self.scaler.normalize(observation)
            states.append(observation)
            action = self.choose_action(observation)
            actions.append(action)
            next_observation, reward, done, info = self.env.step(action)
            rewards.append(reward)
            observation = next_observation

            if Config.train.get('max_episode_steps', None):
                count += 1
                if count == Config.train.max_episode_steps:
                    break

        self.scaler.update(np.array(unscaled_states))

        return states, actions, rewards, next_observation, done

    def cal_target_v(self, done, next_observation, rewards):
        if done:
            next_value = 0
        else:
            next_value = \
            self.sess.run(self.critic.value, {self.critic.states: [self.scaler.normalize(next_observation)]})[0, 0]
        target_v = []
        for reward in rewards[::-1]:
            next_value = reward + Config.train.reward_decay * next_value
            target_v.append([next_value])
        target_v.reverse()
        return target_v

    def update_actor(self, states, actions, target_v):
        feed_dict = {self.critic.states: states, self.actor.states: states}
        value, old_mean, old_stdd = self.sess.run([self.critic.value, self.actor.mean, self.actor.stdd], feed_dict)
        advantage = np.array(target_v) - value
        feed_dict = {self.actor.states: states,
                     self.actor.advantage: advantage,
                     self.actor.old_mean: old_mean,
                     self.actor.old_stdd: old_stdd,
                     self.actor.actions: actions}
        if Config.train.surrogate_clip:
            for e in range(Config.train.actor_train_episode):
                self.sess.run(self.actor.train_op, feed_dict)
        else:
            for e in range(Config.train.actor_train_episode):
                _, kl = self.sess.run([self.actor.train_op, self.actor.kl], feed_dict)
                if kl > Config.train.kl_target * 4:
                    break

            if kl > Config.train.kl_target * Config.train.kl_target_beta:
                Config.train.kl_loss_lam *= Config.train.kl_lam_alpha
            elif kl < Config.train.kl_target / Config.train.kl_target_beta:
                Config.train.kl_loss_lam /= Config.train.kl_lam_alpha

    def update_critic(self, states, target_v):
        feed_dict = {self.critic.states: states, self.critic.target_v: target_v}
        for e in range(Config.train.critic_train_episode):
            self.sess.run(self.critic.train_op, feed_dict)
