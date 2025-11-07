import env_dicts
from modules.config import PPOConfig, wrap_config_from_dict
from runner.run import train_agent

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # safe wrapping
    cfg = wrap_config_from_dict(PPOConfig(), env_dicts.MountainCar_v0_dict)
    # train a new agent under above configurations
    train_agent(cfg)


