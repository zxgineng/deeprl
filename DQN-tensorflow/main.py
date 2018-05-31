import gym
import argparse
import tensorflow as tf

from agent import Agent
from utils import Config


def main(mode):
    env = gym.make(Config.data.environment_name)
    env = env.unwrapped
    num_action = env.action_space.n
    state_dim = env.observation_space.shape[0]
    agent = Agent(num_action, state_dim)

    for episode in range(100):
        observation = env.reset()
        ep_r = 0
        while True:
            env.render()
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)

            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = next_observation
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            agent.store_trainsition(observation,action,reward,next_observation)
            ep_r += reward
            if len(agent.replay_memory)>=Config.train.replay_size:
                agent.train()

            if done:
                print('episode: ', episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(agent.epsilon, 2))
                break





if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='train', choices=['train'],
                        help='Mode (train)')
    parser.add_argument('--config', type=str, default='config/DQN.yml', help='config file name')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    Config(args.config)
    print(Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    main(args.mode)
