
from modules.config import *
import tyro
from runner.run import train_agent

def train_ppo(cfg = tyro.cli(PPOConfig), model_path=None):
    # override cfg
    cfg.algorithm = 'ppo'
    cfg.is_discrete = True
    cfg.env_name = 'LunarLander-v3'

    # train
    train_agent(cfg, model_path=model_path)


def train_dqn(cfg = tyro.cli(DQNConfig),model_path=None):
    # override cfg
    cfg.algorithm = 'dqn'
    cfg.is_discrete = True
    cfg.env_name = 'LunarLander-v3'

    # train
    train_agent(cfg,model_path=model_path)








