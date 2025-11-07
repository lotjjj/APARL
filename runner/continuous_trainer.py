import env_dicts
from modules.config import PPOConfig, wrap_config_from_dict
from runner.run import train_agent

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # safe wrapping
    cfg = wrap_config_from_dict(PPOConfig(), env_dicts.BipedalWalker_v3_dict)

    # cfg.print_info()

    train_agent(cfg)


