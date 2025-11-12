from runner import train_dicts
from modules.config import PPOConfig, wrap_config_from_dict
from runner.run import train_agent, make_vec_env, make_eval_env

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # safe wrapping
    cfg = wrap_config_from_dict(PPOConfig(), train_dicts.BipedalWalker_v3_dict)

    # make train env
    envs = make_vec_env(cfg, wrappers=[])

    # make eval env
    eval_env = make_eval_env(cfg)

    # train a new agent under above configurations
    train_agent(envs, eval_env, cfg)