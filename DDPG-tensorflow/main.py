import argparse
import tensorflow as tf
import gym

from agent import Agent
from utils import Config


def run(mode, run_config, params):
    env = gym.make(Config.data.env_name).unwrapped
    agent = Agent(env)
    estimator = tf.estimator.Estimator(
        model_fn=agent.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)

    def input():
        inputs = tf.placeholder(tf.float32,
                                [None, Config.data.state_dim], 'current_state')

        return inputs, None

    if mode == 'train':
        estimator.train(input_fn=input, max_steps=Config.train.max_steps)
    elif mode == 'eval':
        estimator.evaluate(input_fn=input)
    env.close()


def main(mode):
    params = tf.contrib.training.HParams(**Config.train.to_dict())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        model_dir=Config.train.model_dir,
        session_config=config,
        save_checkpoints_steps=Config.train.save_checkpoints_steps,
        log_step_count_steps=None)

    run(mode, run_config, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='train', choices=['train','eval'],
                        help='Mode (train)')
    parser.add_argument('--config', type=str, default='config/ddpg.yml', help='config file name')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    Config(args.config)
    print(Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    main(args.mode)
