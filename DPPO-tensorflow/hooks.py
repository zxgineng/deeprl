import tensorflow as tf
import os
from collections import deque

from utils import Config,save_training_state,load_training_state


class TrainHook(tf.train.SessionRunHook):
    def __init__(self,agent):
        self.env = agent.env
        self.agent = agent

        # self.done=True
        self.ep_reward_queue = deque([],100)

        if not os.path.exists(Config.data.base_path):
            os.makedirs(Config.data.base_path)

    def _load_training_state(self):
        training_state = load_training_state()
        if training_state:
            self.agent.replay_memory.load_memory(training_state['memory'])
            self.episode = training_state['episode']
            self.ep_reward_queue = training_state['reward_queue']
            print('training state loaded.')

    def _save_training_state(self):
        save_training_state(memory=self.agent.replay_memory.get_memory(),episode=self.episode,reward_queue=self.ep_reward_queue)
        print('training state saved.')

    def after_create_session(self, session, coord):
        self.agent.sess = session
        self._load_training_state()

    def before_run(self,run_context):

        observation = self.env.reset()
        ep_reward = 0


        for step in range(200):
            buffer_states, buffer_actions, buffer_rewards = [], [], []
            action = self.agent.choose_action(observation)
            next_observation, reward, done, info = self.env.step(action)

            buffer_states.append(observation)
            buffer_actions.append(action)
            buffer_rewards.append(reward)

            observation =next_observation
            ep_reward += reward

            if (step+1) % Config.train.batch_size == 0 or done or step == 200 -1:
                if done:
                    next_value = 0
                else:
                    next_value = self.sess.run('value:0',{'states:0':[next_observation]})[0,0]

                buffer_v_target = []
                for reward in buffer_rewards[::-1]:
                    next_value = reward + Config.train.reward_decay * next_value
                    buffer_v_target.append([next_value])
                buffer_v_target.reverse()
                self.agent.learn(buffer_states,buffer_actions,buffer_rewards)

            if done or step == 200-1:
                self.ep_reward_queue.append(ep_reward)
                ave_ep_reward = round(sum(self.ep_reward_queue) / len(self.ep_reward_queue), 2)
                return tf.train.SessionRunArgs('global_step:0',
                                               feed_dict={'ave_ep_reward:0': ave_ep_reward})






