import tensorflow as tf
import numpy as np

from utils import Config
from architecture import Graph
from replay_memory import ExperienceReplay

class Agent():

    def __init__(self,num_action,state_dim):
        self.num_action = num_action
        self.state_dim = state_dim
        self.train_step = 0
        self.epsilon = Config.train.initial_epsilon

        self.build_model()
        self.build_loss()
        self.build_train_op()

        self.replay_memory = ExperienceReplay(Config.train.replay_size)

        train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'train_net')
        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_net')
        self.replace_target_op = [tf.assign(target_param,train_param) for train_param,target_param in zip(train_params,target_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        train_graph = Graph('train_net')
        target_graph = Graph('target_net')
        self.state = tf.placeholder(tf.float32,[None,self.state_dim])
        self.next_state = tf.placeholder(tf.float32,[Config.train.batch_size,self.state_dim])

        self.q_eval = train_graph.build(self.state,self.num_action)
        self.q_next = target_graph.build(self.state,self.num_action)

    def build_loss(self):
        y = tf.placeholder(tf.float32,[Config.train.batch_size,self.num_action],'q_target')
        self.loss = tf.reduce_mean(tf.squared_difference(y, self.q_eval))

    def build_train_op(self):
        self.train_op = tf.train.RMSPropOptimizer(Config.train.learning_rate).minimize(self.loss)

    def choose_action(self,observation):
        observation = np.expand_dims(observation,0)

        if np.random.uniform() >= self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,self.num_action)
        return action

    def store_trainsition(self,observation,action,reward,next_observation):
        self.replay_memory.add((observation,action,reward,next_observation))

    def train(self):
        # check to replace target parameters
        if self.train_step % Config.train.replace_target_n_iter == 0:
            self.sess.run(self.replace_target_op)
            print('target params replaced')
        batch = self.replay_memory.get_batch(Config.train.batch_size)
        state = batch[:,:self.state_dim]
        next_state = batch[:,-self.state_dim:]
        q_next,q_eval = self.sess.run([self.q_next,self.q_eval],feed_dict={self.next_state:next_state,self.state:state})

        q_target = q_eval.copy()
        eval_action_index = batch[:,self.state_dim].astype(int)
        reward = batch[:,self.state_dim+1]
        q_target[np.arange(Config.train.batch_size),eval_action_index] = reward + Config.train.reward_decay * np.max(q_next,-1)
        _,loss = self.sess.run([self.train_op,self.loss],feed_dict={self.state:state,'q_target:0':q_target})
        self.epsilon = max(self.epsilon - Config.train.epsilon_decrease,0)
        self.train_step+=1

