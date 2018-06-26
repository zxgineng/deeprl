import numpy as np
import tensorflow as tf

from hooks import TrainingHook, EvalHook
from actor import Actor
from critic import Critic
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
        self.ou_noise = OUNoise(Config.data.action_dim, Config.train.noise_theta, Config.train.noise_sigma,
                                Config.train.noise_mu)
        with tf.variable_scope('eval_net'):
            self.eval_actor = Actor()
            self.eval_critic = Critic(self.eval_actor.actions)

            self.eval_actor.build_train_op(self.eval_critic.qa_value)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_update_op()
            ave_ep_reward = tf.placeholder(tf.float32, None, name='ave_ep_reward')
            tf.summary.scalar('ave_reward', ave_ep_reward)
            self.loss = ave_ep_reward
            self.train_op = tf.no_op()
            self.training_hooks = [TrainingHook(self)]
        else:
            self.loss = tf.constant(0)
            self.evaluation_hooks = [EvalHook(self)]

    def _build_update_op(self):
        global_step = tf.train.get_global_step()
        tf.assign_add(global_step, 1, name='global_step_add')

        # with tf.variable_scope('eval_net'):
        #     self.eval_critic = Critic(self.eval_actor.actions)
        #
        #     self.eval_actor.build_train_op(self.eval_critic.qa_value)

        with tf.variable_scope('target_net'):
            self.target_actor = Actor()
            self.target_critic = Critic(self.target_actor.actions)

        actor_update_op = [
            tf.assign(target_param, target_param * (1 - Config.train.TAU) + train_param * Config.train.TAU)
            for train_param, target_param in zip(self.eval_actor.params, self.target_actor.params)]

        critic_update_op = [
            tf.assign(target_param, target_param * (1 - Config.train.TAU) + train_param * Config.train.TAU)
            for train_param, target_param in zip(self.eval_critic.params, self.target_critic.params)]

        actor_init_op = [tf.assign(target_param, train_param) for train_param, target_param in
                         zip(self.eval_actor.params, self.target_actor.params)]

        critic_init_op = [tf.assign(target_param, train_param) for train_param, target_param in
                          zip(self.eval_critic.params, self.target_critic.params)]

        self.update_target_op = tf.group(actor_update_op + critic_update_op)
        self.init_target_op = tf.group(actor_init_op + critic_init_op)

    def update_params(self):
        batch = self.replay_memory.get_batch(Config.train.batch_size)
        states, actions, rewards, next_states = [], [], [], []

        for d in batch:
            states.append(d[0])
            actions.append(d[1])
            rewards.append(d[2])
            next_states.append(d[3])

        next_qa = self.sess.run(self.target_critic.qa_value,
                                {self.target_critic.states: next_states, self.target_actor.states: next_states})

        target_qa = []

        for i in range(len(batch)):
            terminal = batch[i][4]
            # if terminal, only equals reward
            if terminal:
                target_qa.append([rewards[i]])
            else:
                target_qa.append([rewards[i] + Config.train.reward_decay * next_qa[i][0]])

        self.sess.run(self.eval_critic.train_op,
                      {self.eval_critic.states: states, self.eval_critic.actions: actions,
                       self.eval_critic.target_qa: target_qa})

        self.sess.run([self.eval_actor.train_op, 'global_step_add:0'],
                      {self.eval_actor.states: states, self.eval_critic.states: states})

    def update_target_params(self):
        self.sess.run(self.update_target_op)

    def init_target_params(self):
        self.sess.run(self.init_target_op)

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

            if Config.train.get('max_episode_steps', None):
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

            ep_reward += reward
            self.store_transition(observation, action, reward, next_observation, done)
            observation = next_observation

            if self.replay_memory.length >= Config.train.observe_n_iter:
                self.update_params()
                self.update_target_params()

            if Config.train.get('max_episode_steps', None):
                count += 1
                if count == Config.train.max_episode_steps:
                    break
        return ep_reward

    def choose_action(self, observation, noise=True):
        action = self.sess.run(self.eval_actor.actions, feed_dict={self.eval_actor.states: [observation]})[0]
        if noise:
            noise = self.ou_noise.noise()  # add exploration noise
            action = action + noise
        return action

    def store_transition(self, *args):
        self.replay_memory.add(args)


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
