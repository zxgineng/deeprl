# Reinforce-PG

TensorFlow implementation of Reinforce Policy Gradient

## Requirements

- Python 3
- TensorFlow 1.5
- gym


## Project Structure


    ├── config                  # Config files (.yml)
    ├── architecture            # architecture graphs
        ├── __init__.py             # network
    ├── agent.py                # define agent, model, loss
    ├── main.py                 # train and evaluate
    ├── utils.py                # config tools 
    └── hooks.py                # define hooks
    

## Config

reinforce.yml

```yml
data:
  base_path: 'data/'
  save_state_file: 'state.pkl'
  env_name: 'MountainCar-v0'

train:
  reward_decay: 0.995

  learning_rate: 0.01
  save_checkpoints_steps: 100
  model_dir: 'logs/reinforce'
  max_steps: 1000

```


## Run


Train

```
python main.py
```

