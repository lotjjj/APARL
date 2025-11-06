# APA RL


## Basic usage


### Config
A custom configuration can be setup by create a custom class that inherit 'Class PPOConfig' or any other algorithm's dataclass, or you can inherit 'BasicConfig' and define your algorithm.

If you want to apply PPO in two different scenarios with different algorithm configuration, 
- Create different configuration class inherit PPOConfig and @override some parameters
- [Not Implemented] import configuration from yaml/json file.


