import numpy as np
import tensorflow as tf

from replay_memory import PriorizedExperienceReplay
from utils import Config


class Agent:

    def __init__(self):
        self.replay_memory = PriorizedExperienceReplay(Config.train.memory_size)

    def choose_action(self, observation, sess, epsilon, one_hot=False):

        observation = np.expand_dims(observation, 0)

        if np.random.uniform() >= epsilon:
            actions_value = sess.run('q_eval:0', feed_dict={'current_state:0': observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, Config.data.num_action)

        if one_hot:
            action_index = action
            action = np.zeros(Config.data.num_action)
            action[action_index] = 1
        return action

    def learn(self, sess):
        self.leaf_idx, batch, ISWeights = self.replay_memory.get_batch(Config.train.batch_size)

        state, action, reward, next_state = [], [], [], []
        for d in batch:
            state.append(d[0])
            if not isinstance(d[1],(int,np.int64)):
                action.append(d[1])
            else:
                one_hot = [0] * Config.data.num_action
                one_hot[d[1]] = 1
                action.append(one_hot)
            reward.append(d[2])
            next_state.append(d[3])

        y = []
        q_next, q_eval_next = sess.run(['q_next:0', 'q_eval:0'],
                                       feed_dict={'next_state:0': next_state, 'current_state:0': next_state})

        for i in range(len(batch)):
            terminal = batch[i][4]
            # if terminal, only equals reward
            if terminal:
                y.append(reward[i])
            else:
                a_next_index = np.argmax(q_eval_next, -1)
                y.append(reward[i] + Config.train.reward_decay * q_next[i, a_next_index[i]])

        return tf.train.SessionRunArgs(['global_step:0','td_errors:0'],
                                       feed_dict={'q_target:0': y, 'current_state:0': state, 'action:0': action, 'ISWeights:0':ISWeights})
