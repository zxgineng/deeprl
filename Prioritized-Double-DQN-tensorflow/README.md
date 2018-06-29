# Prioritized Double DQN

TensorFlow implementation of [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf). (2015. 11)

![images](images/paper1.png)

## Requirements

- Python 3
- TensorFlow 1.5
- gym


## Project Structure


    ├── config                  # Config files (.yml)
    ├── model.py                # network, loss
    ├── agent.py                # agent 
    ├── main.py                 # train and eval
    ├── utils.py                # config, save tools
    ├── replay_memory.py        # restore and sample 
    └── hooks.py                # train and eval hooks
    

## Config

double-DQN.yml

```yml
data:
  base_path: 'data/'
  save_state_file: 'state.pkl'
  env_name: 'MountainCar-v0'

train:
  batch_size: 32

  initial_epsilon: 1.0
  epsilon_decrement: 0.0002
  final_epsilon: 0.1

  reward_decay: 0.9
  observe_n_iter: 1500
  memory_size: 3000
  replace_target_n_iter: 300

  learning_rate: 0.001
  save_checkpoints_steps: 10000
  model_dir: 'logs/double-DQN'
  max_steps: 70000
```


## Run


Train

```
python main.py --mode train
```

Evaluate

```
python main.py --mode eval
```

## Tensorboard
Average reward of 100 episode

![images](images/ave-ep-reward.png)


## Example
Teach the car to drive up a big hill

![images](images/example.gif)
