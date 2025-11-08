# APA RL


## Basic usage


### Config
A custom configuration can be setup by create a custom class that inherit 'Class PPOConfig' or any other algorithm's dataclass, or you can inherit 'BasicConfig' and define your algorithm.

But it is strongly suggested to use env_dicts and function: wrap_config_from _dict, or import your configuration from file.

### Not Implemented
- Bug fixes: Wrongly remove
- Tensorboard
- Graph
- DQN
- Learning rate scheduler
- PPO: works not well

