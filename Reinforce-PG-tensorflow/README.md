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
    ├── agent.py                # agent
    ├── main.py                 # train and evaluate
    ├── utils.py                # config tools 
    └── model.py                # define model, loss, algo
    

## Config

reinforce.yml

```yml
data:
  base_path: 'data/'
  save_state_file: 'state.pkl'
  num_action: 3
  state_dim: 2

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

