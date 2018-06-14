import tensorflow as tf
import os

from utils import Config, save_training_state, load_training_state


class TrainHook(tf.train.SessionRunHook):

    def __init__(self, agent):
        self.env = agent.env
        self.agent = agent

        self.episode = 0
        self.done = True

        self.running_reward = 0

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.episode = training_state['episode']
            print('episode: ', self.episode, '    training state loaded.')

    def _save_training_state(self):
        save_training_state(episode=self.episode)
        print('episode: ', self.episode, '    training state saved.')

    def after_create_session(self, session, coord):
        self.agent.sess = session
        self._load_training_state()

    def before_run(self, run_context):

        while True:
            if self.done:
                self.observation = self.env.reset()
            # self.env.render()

            action = self.agent.choose_action(self.observation)

            next_observation, reward, self.done, info = self.env.step(action)

            self.agent.store_transition(self.observation, action, reward)

            self.observation = next_observation

            if self.done:
                ep_reward = sum(self.agent.ep_reward)

                self.episode += 1
                # for smooth reward metric
                if self.episode == 1:
                    self.running_reward = ep_reward
                else:
                    self.running_reward = self.running_reward * 0.99 + ep_reward * 0.01

                print('episode: ', self.episode, '  running_reward: ', round(self.running_reward, 2))

                return self.agent.learn()  # learn at the end of each episode

    def after_run(self, run_context, run_values):
        self.agent.ep_reset()

        global_step = run_values.results

        if global_step == 1 or global_step % Config.train.save_checkpoints_steps == 0:
            self._save_training_state()

    def end(self, session):
        self._save_training_state()
