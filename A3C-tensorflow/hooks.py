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
        # self.eval_count = 0
        # self.ave_ep_reward_saver = []
        self.start = False
        self.last_step = 0

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.ep_reward_queue = training_state['reward_queue']
            # self.eval_count = training_state['eval_count']
            # self.ave_ep_reward_saver = training_state['ave_ep_reward_saver']
            self.last_step = training_state['last_step']
            print('training state loaded.')

    def _save_training_state(self):
        save_training_state(reward_queue=self.ep_reward_queue,
                            # eval_count=self.eval_count, ave_ep_reward_saver=self.ave_ep_reward_saver,
                            last_step = self.last_step)
        print('training state saved.')

    def after_create_session(self, session, coord):
        self.chief.sess = session
        for worker in self.workers:
            worker.sess = session
            worker.coord = coord
        self._load_training_state()

    def before_run(self, run_context):

        if not self.start:
            for worker in self.workers:
                t = threading.Thread(target=worker.learn)
                t.start()
            self.start = True

        ep_reward = self.chief.eval()
        # self.eval_count += 1
        self.ep_reward_queue.append(ep_reward)
        ave_ep_reward = round(sum(self.ep_reward_queue) / len(self.ep_reward_queue), 2)
        # if self.eval_count % 100 == 0:
        #     print('episode:',self.eval_count,'  ave ep reward:', ave_ep_reward)
        #     self.ave_ep_reward_saver.append((Config.train.global_ep, ave_ep_reward))

        return tf.train.SessionRunArgs('global_step:0',
                                       feed_dict={'ave_ep_reward:0': ave_ep_reward})

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        # synchronized with checkpoints
        if global_step - self.last_step >= Config.train.save_checkpoints_steps:
            self.last_step = global_step
            self._save_training_state()

    def end(self, session):
        self._save_training_state()
