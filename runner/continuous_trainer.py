import env_dicts
from modules.config import PPOConfig, wrap_config_from_dict
from runner.run import train_agent

if __name__ == '__main__':
    # safe wrapping
    cfg = wrap_config_from_dict(PPOConfig(), env_dicts.MountainCar_v0_dict)

    # cfg.print_info()

    train_agent(cfg)


