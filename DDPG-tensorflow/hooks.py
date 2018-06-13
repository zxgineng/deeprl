import tensorflow as tf
from collections import deque

from utils import *


class TrainHook(tf.train.SessionRunHook):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.done = True
        self.episode = 0
        self.ep_reward_queue = deque([], 100)
        self.ave_reward_list = []

        self._build_replace_target_op()

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.agent.replay_memory.load_memory(training_state['memory'])
            self.episode = training_state['episode']
            self.ep_reward_queue = training_state['reward_queue']
            self.ave_reward_list = training_state['ave_reward_list']
            print('episode: ', self.episode, '    training state loaded.')

    def _save_training_state(self):
        save_training_state(memory=self.agent.replay_memory.get_memory(), episode=self.episode,
                            reward_queue=self.ep_reward_queue, ave_reward_list=self.ave_reward_list)
        print('episode: ', self.episode, '    training state saved.')

    def _build_replace_target_op(self):
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor/eval')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic/eval')
        at_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor/target')
        ct_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic/target')

        actor_replace_op = [tf.assign(target_param, target_param * (
                1 - Config.train.TAU) + train_param * Config.train.TAU) for
                            train_param, target_param in
                            zip(a_params, at_params)]
        critic_replace_op = [tf.assign(target_param, target_param * (
                1 - Config.train.TAU) + train_param * Config.train.TAU) for
                             train_param, target_param in
                             zip(c_params, ct_params)]

        actor_initial_op = [tf.assign(target_param, train_param) for
                            train_param, target_param in
                            zip(a_params, at_params)]
        critic_initial_op = [tf.assign(target_param, train_param) for
                             train_param, target_param in
                             zip(c_params, ct_params)]

        self.initial_replace_op = actor_initial_op + critic_initial_op
        self.replace_target_op = actor_replace_op + critic_replace_op

    def after_create_session(self, session, coord):
        self.agent.sess = session
        session.run(self.initial_replace_op)
        self._load_training_state()

    def before_run(self, run_context):

        while True:
            if self.done:
                self.observation = self.env.reset()
                self.ep_reward = 0

            # self.env.render()
            action = self.agent.choose_action(self.observation)

            next_observation, reward, self.done, info = self.env.step(action)

            self.agent.store_transition(self.observation, action, reward, next_observation, self.done)

            self.ep_reward += reward

            if self.done:
                self.episode += 1
                self.ep_reward_queue.append(self.ep_reward)
                if self.episode % 100 == 0:
                    ave_ep_reward = round(sum(self.ep_reward_queue) / len(self.ep_reward_queue),2)
                    self.ave_reward_list.append(ave_ep_reward)
                    print('*' * 40)
                    print('episode:', self.episode, ' ave_ep_reward:', ave_ep_reward)
                    print('*' * 40)
            else:
                self.observation = next_observation

            if self.agent.replay_memory.length >= Config.train.observe_n_iter:
                return self.agent.learn()

    def after_run(self, run_context, run_values):

        global_step = run_values.results
        run_context.session.run(self.replace_target_op)

        if global_step == 1 or global_step % Config.train.save_checkpoints_steps == 0:
            self._save_training_state()

    def end(self, session):
        self._save_training_state()


class EvalHook(tf.train.SessionRunHook):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.done = True
        self.episode = 0

    def after_create_session(self, session, coord):
        self.agent.sess = session

    def before_run(self, run_context):
        while True:
            if self.done:
                self.observation = self.env.reset()
                self.ep_reward = 0

            self.env.render()
            action = self.agent.choose_action(self.observation,noise=False)
            next_observation, reward, self.done, info = self.env.step(action)
            self.ep_reward += reward

            if self.done:
                self.episode += 1
                print('episode:', self.episode, ' ep_reward:', self.ep_reward)

            else:
                self.observation = next_observation

