
from modules.config import *
import tyro
from runner.run import train_agent


class LunarLanderConfig(PPOConfig, DQNConfig):
    pass

def train_ppo(cfg = tyro.cli(PPOConfig)):
    # override cfg
    cfg.algorithm = 'ppo'
    cfg.is_discrete = True
    cfg.env_name = 'LunarLander-v3'

    # train
    train_agent(cfg)


def train_dqn(cfg = tyro.cli(DQNConfig)):
    # override cfg
    cfg.algorithm = 'dqn'
    cfg.is_discrete = True

    # train
    train_agent(cfg)







