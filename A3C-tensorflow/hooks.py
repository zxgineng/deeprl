import tensorflow as tf
import os
from collections import deque
import threading

from utils import Config, save_training_state, load_training_state


class TrainingHook(tf.train.SessionRunHook):
    def __init__(self, chief, workers):
        self.workers = workers
        self.chief = chief
        self.ep_reward_queue = deque([], 100)
        self.start = True
        self.last_step = 0
        self.first_save = True

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.chief.scaler = training_state['scaler']
            self.last_step = training_state['last_step']
            self.ep_reward_queue = training_state['ep_reward_queue']
            print('training state loaded.')

    def _save_training_state(self):
        save_training_state(scaler=self.chief.scaler, last_step=self.last_step, ep_reward_queue=self.ep_reward_queue)
        print('training state saved.')

    def after_create_session(self, session, coord):

        self.chief.sess = session
        for worker in self.workers:
            worker.sess = session
            worker.coord = coord
        self.chief.init_scaler()
        self._load_training_state()

    def before_run(self, run_context):
        if self.start:
            for worker in self.workers:
                t = threading.Thread(target=worker.work)
                t.start()
            self.start = False

        ep_reward = self.chief.eval()
        self.ep_reward_queue.append(ep_reward)
        ave_ep_reward = round(sum(self.ep_reward_queue) / len(self.ep_reward_queue), 2)

        return tf.train.SessionRunArgs('global_step:0',
                                       feed_dict={'ave_ep_reward:0': ave_ep_reward})

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
        self.env = agent.env
        self.agent = agent

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.agent.scaler = training_state['scaler']
            print('training state loaded.')

    def after_create_session(self, session, coord):
        self.agent.sess = session
        self._load_training_state()

    def before_run(self, run_context):
        ep_reward = self.agent.eval(True)
        print('ep_reward:', ep_reward)
