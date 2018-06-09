import tensorflow as tf

from utils import *


class AlgoTrainHook(tf.train.SessionRunHook):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.done = True

        self._build_replace_target_op()

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

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
        file = os.path.join(Config.data.base_path, Config.data.save_state_file)
        try:
            if os.path.isfile(file):
                training_state = load_training_state()
                self.agent.replay_memory.load_memory(training_state['memory'])
                print('training state loaded:')
                print('replay memory: %d' % (self.agent.replay_memory.length))
        except Exception as e:
            print(e.args[0])
            print('load state failed.')

    def before_run(self, run_context):

        while True:
            if self.done:
                self.observation = self.env.reset()

            action = self.agent.choose_action(self.observation)

            next_observation, reward, self.done, info = self.env.step(action)

            self.agent.store_transition(self.observation, action, reward, next_observation)
            self.observation = next_observation

            if self.agent.replay_memory.length >= Config.train.observe_n_iter:
                return self.agent.learn()

    def after_run(self, run_context, run_values):

        global_step = run_values.results
        run_context.session.run(self.replace_target_op)

        if global_step == 0 or global_step % Config.train.save_checkpoints_steps == 0:
            try:
                save_training_state(memory=self.agent.replay_memory.get_memory())
                print('training state saved.')
            except Exception as e:
                print(e.args[0])
                print('save state failed.')

    def end(self, session):
        try:
            save_training_state(memory=self.agent.replay_memory.get_memory())
            print('training state saved.')
        except Exception as e:
            print(e.args[0])
            print('save state failed.')
