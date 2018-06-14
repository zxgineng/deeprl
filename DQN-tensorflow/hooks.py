import tensorflow as tf
import os

from utils import Config,save_training_state,load_training_state


class TrainHook(tf.train.SessionRunHook):

    def __init__(self, agent):
        self.env = agent.env
        self.agent = agent

        self.episode = 0
        self.done = True
        self._build_replace_target_op()

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.agent.replay_memory.load_memory(training_state['memory'])
            self.episode = training_state['episode']
            print('episode: ', self.episode, '    training state loaded.')

    def _save_training_state(self):
        save_training_state(memory=self.agent.replay_memory.get_memory(), episode=self.episode)
        print('episode: ', self.episode, '    training state saved.')

    def _build_replace_target_op(self):
        train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'eval_net')
        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_net')
        self.replace_target_op = [tf.assign(target_param, train_param) for train_param, target_param in
                                  zip(train_params, target_params)]

    def after_create_session(self, session, coord):
        self.agent.sess = session
        self.epsilon = Config.train.initial_epsilon
        self._load_training_state()

    def before_run(self, run_context):

        while True:
            if self.done:
                self.observation = self.env.reset()
                self.ep_reward = 0

            self.env.render()

            action = self.agent.choose_action(self.observation, self.epsilon)

            next_observation, reward, self.done, info = self.env.step(action)

            reward = abs(next_observation[0] - (-0.5))

            self.agent.store_transition(self.observation, action, reward, next_observation, self.done)

            self.ep_reward += reward

            self.observation = next_observation

            if self.done:
                self.episode += 1
                print('episode: ', self.episode, '  ep_reward: ', round(self.ep_reward, 2))

            if self.agent.replay_memory.get_length() >= Config.train.observe_n_iter:
                return self.agent.learn()
            else:
                continue

    def after_run(self, run_context, run_values):

        global_step = run_values.results
        if global_step % Config.train.replace_target_n_iter == 0:
            run_context.session.run(self.replace_target_op)
            print('target params replaced.')

        if global_step ==1 or global_step % Config.train.save_checkpoints_steps == 0:
            self._save_training_state()

    def end(self, session):
        self._save_training_state()