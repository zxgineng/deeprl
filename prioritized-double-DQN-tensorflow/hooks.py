import tensorflow as tf
import os
from collections import deque

from utils import Config, save_training_state, load_training_state


class TrainingHook(tf.train.SessionRunHook):
    def __init__(self, agent):
        self.agent = agent
        self.ep_reward_queue = deque([], 100)
        self.last_step = 0
        self.first_save = True

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.agent.replay_memory.load_memory(training_state['replay_memory'])
            self.last_step = training_state['last_step']
            Config.train.epsilon = training_state['epsilon']
            print('training state loaded.')

    def _save_training_state(self):
        save_training_state(epsilon=Config.train.epsilon,replay_memory=self.agent.replay_memory.get_memory(), last_step=self.last_step)
        print('training state saved.')

    def after_create_session(self, session, coord):
        self.agent.sess = session
        self.agent.replace_target_params()
        Config.train.epsilon = Config.train.initial_epsilon
        self._load_training_state()

    def before_run(self, run_context):
        ep_reward = self.agent.run_episode()
        self.ep_reward_queue.append(ep_reward)
        ave_ep_reward = sum(self.ep_reward_queue) / len(self.ep_reward_queue)
        return tf.train.SessionRunArgs('global_step:0', feed_dict={'ave_ep_reward:0': ave_ep_reward})

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        # synchronized with checkpoints
        if self.first_save or (global_step - self.last_step >= Config.train.save_checkpoints_steps):
            self.first_save = False
            self.last_step = global_step
            self._save_training_state()

    def end(self, session):
        self._save_training_state()


class EvalHook(tf.train.SessionRunHook):
    def __init__(self, agent):
        self.agent = agent

    def after_create_session(self, session, coord):
        self.agent.sess = session
        Config.train.epsilon = 0.1

    def before_run(self, run_context):
        ep_reward = self.agent.eval(True)
        print('ep_reward:',ep_reward)