import tensorflow as tf
from tensorflow.contrib import slim
import os
import gym

from utils import Config,save_training_state,load_training_state
from architecture import Graph
from agent import Agent


class Model:

    def __init__(self):
        self.env = gym.make('MountainCar-v0').unwrapped
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
        graph = Graph()
        logits = graph.build(self.state)
        tf.nn.softmax(logits,-1,'action_prob')

        # self.predictions = q_eval

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits)
            self._build_train_op()

    def _build_loss(self,logits):
        action = tf.placeholder(tf.int32, [None], 'action')
        vt = tf.placeholder(tf.float32, [None],'action_value')
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action,logits=logits) * vt)

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, self.global_step,
            optimizer=tf.train.AdamOptimizer(Config.train.learning_rate),
            learning_rate=Config.train.learning_rate,
            name="train_op")

        class AlgoTrainHook(tf.train.SessionRunHook):

            def __init__(self, env, agent):
                self.env = env
                self.agent = agent

                self.episode = 0
                self.done = True

                self.running_reward = 0

                if not os.path.exists(Config.data.base_path):
                    os.makedirs(Config.data.base_path)

            def after_create_session(self, session, coord):
                self.agent.sess = session

                file = os.path.join(Config.data.base_path, Config.data.save_state_file)
                try:
                    if os.path.isfile(file):
                        training_state = load_training_state()
                        self.episode = training_state['episode']

                        print('training state loaded:')
                        print('episode: %d'%(self.episode))
                except Exception as e:
                    print(e.args[0])
                    print('load state failed.')

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

                        return self.agent.learn()    # learn at the end of each episode

            def after_run(self, run_context, run_values):
                self.agent.ep_reset()

                global_step = run_values.results

                if global_step % Config.train.save_checkpoints_steps == 0:
                    try:
                        save_training_state(episode=self.episode)
                        print('training state saved.')
                    except Exception as e:
                        print(e.args[0])
                        print('save state failed.')

        self.training_hooks = [AlgoTrainHook(self.env, self.agent)]