import tensorflow as tf
import os
from collections import deque

from utils import Config, save_training_state, load_training_state


class TrainingHook(tf.train.SessionRunHook):
    def __init__(self, agent):
        self.env = agent.env
        self.agent = agent

        self.ep_reward_queue = deque([], 100)

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.agent.scaler = training_state['scaler']
            print('training state loaded.')

    def _save_training_state(self):
        save_training_state(scaler=self.agent.scaler)
        print('training state saved.')

    def after_create_session(self, session, coord):
        self.agent.sess = session
        self.agent.init_scaler()
        self._load_training_state()

    def before_run(self, run_context):
        states, actions, rewards = self.agent.run_episode()
        next_value = 0
        target_v = []
        for reward in rewards[::-1]:
            next_value = reward + Config.train.reward_decay * next_value
            target_v.append([next_value])
        target_v.reverse()
        self.agent.update_actor(states, actions, target_v)
        self.agent.update_critic(states, target_v)

        self.ep_reward_queue.append(sum(rewards))
        ave_ep_reward = sum(self.ep_reward_queue) / len(self.ep_reward_queue)

        return tf.train.SessionRunArgs('global_step:0', feed_dict={'ave_ep_reward:0': ave_ep_reward})

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if global_step == 1 or global_step % Config.train.save_checkpoints_steps == 0:
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
