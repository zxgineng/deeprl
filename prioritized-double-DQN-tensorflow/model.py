import tensorflow as tf
from tensorflow.contrib import slim
import os
import gym

from utils import Config, save_training_state, load_training_state
from architecture import Graph
from agent import Agent


class Model:

    def __init__(self):
        self.env = gym.make('Pendulum-v0').unwrapped
        self.agent = Agent()

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.state = features
        self.loss, self.train_op, self.predictions, self.training_hooks = None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks)

    def build_graph(self):
        graph = Graph('eval_net')

        q_eval = graph.build(self.state)
        tf.identity(q_eval, 'q_eval')

        self.predictions = q_eval

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(q_eval)
            self._build_train_op()

    def _build_loss(self, q_eval):
        graph = Graph('target_net')
        next_state = tf.placeholder(tf.float32,
                                    [Config.train.batch_size, Config.data.state_dim], 'next_state')
        q_next = graph.build(next_state)
        tf.identity(q_next, 'q_next')

        y = tf.placeholder(tf.float32, [None], 'q_target')
        action = tf.placeholder(tf.float32, [None, Config.data.num_action], 'action')
        q_a_eval = tf.reduce_sum(q_eval * action, -1)

        self.loss = tf.reduce_mean(tf.squared_difference(y, q_a_eval))

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, self.global_step,
            optimizer=tf.train.RMSPropOptimizer(Config.train.learning_rate),
            learning_rate=Config.train.learning_rate,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'eval_net'),
            name="train_op")

        class AlgoTrainHook(tf.train.SessionRunHook):

            def __init__(self, env, agent, model):
                self.env = env
                self.agent = agent
                self.model = model

                self.done = True
                self._build_replace_target_op()

                if not os.path.exists(Config.data.base_path):
                    os.makedirs(Config.data.base_path)

            def _build_replace_target_op(self):
                train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'eval_net')
                target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_net')
                self.replace_target_op = [tf.assign(target_param, train_param) for train_param, target_param in
                                          zip(train_params, target_params)]

            def after_create_session(self, session, coord):
                self.epsilon = Config.train.initial_epsilon

                file = os.path.join(Config.data.base_path, Config.data.save_state_file)
                try:
                    if os.path.isfile(file):
                        training_state = load_training_state()
                        self.agent.replay_memory.load_memory(training_state['replay_memory'])

                        print('training state loaded:')
                        print('replay memory: %d' %(self.agent.replay_memory.get_length()))
                except Exception as e:
                    print(e.args[0])
                    print('load state failed.')

            def before_run(self, run_context):

                while True:
                    if self.done:
                        self.observation = self.env.reset()

                    self.env.render()

                    action = self.agent.choose_action(self.observation, run_context.session,
                                                      self.epsilon)
                    f_action = (action - (Config.data.num_action - 1) / 2) / (
                                (Config.data.num_action - 1) / 4)  # convert to [-2 ~ 2] float actions

                    next_observation, reward, self.done, info = self.env.step([f_action])
                    self.agent.replay_memory.add(self.observation, action, reward, next_observation, self.done)

                    self.observation = next_observation

                    if self.agent.replay_memory.get_length() >= Config.train.observe_n_iter:
                        return self.agent.learn(run_context.session)
                    else:
                        continue

            def after_run(self, run_context, run_values):

                global_step = run_values.results
                if global_step % Config.train.replace_target_n_iter == 0:
                    run_context.session.run(self.replace_target_op)
                    print('target params replaced.')

                if global_step % Config.train.save_checkpoints_steps == 0:
                    try:
                        save_training_state(replay_memory=self.agent.replay_memory.get_memory())
                        print('training state saved.')
                    except Exception as e:
                        print(e.args[0])
                        print('save state failed.')

        self.training_hooks = [AlgoTrainHook(self.env, self.agent, self)]
