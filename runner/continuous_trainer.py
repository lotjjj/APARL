import gymnasium as gym
import env_dicts
from modules.config import PPOConfig, wrap_config_from_dict
from runner.run import train_agent, make_vec_env

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # safe wrapping
    cfg = wrap_config_from_dict(PPOConfig(), env_dicts.MountainCarContinuous_v0_dict)

    # make env
    envs = make_vec_env(cfg, wrappers=[gym.wrappers.NormalizeObservation])

    # train a new agent under above configurations
    train_agent(envs, cfg)


