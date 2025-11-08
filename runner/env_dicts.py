

from modules.config import wrap_config_from_dict

LunarLander_v3_dict = {
    'env_name': 'LunarLander-v3',
    'is_discrete': True,

    'horizon_len': 400,
    'num_envs': 6,
    'batch_size': 255,
    'entropy_coef': 0.001,

    'max_episode_steps': 300,
    'eval_max_episode_steps': 400,
    'eval_num_episodes': 3,
    'eval_render_mode': 'None',

    'eval_interval': 50,
    'save_interval': 1000,
}

MountainCarContinuous_v0_dict = {
    'env_name': 'MountainCarContinuous-v0',
    'is_discrete': False,

    'device': 'cpu',

    'num_envs': 6,
    'batch_size': 128,

    'eval_render_mode': 'None',

    'options': {"low": -1, "high": -0.5},
}

BipedalWalker_v3_dict = {
    'env_name': 'BipedalWalker-v3',
    'is_discrete': False,

    'horizon_len': 600,
    'num_envs': 3,
    'batch_size': 123,
    'entropy_coef': 0.001,

    'max_episode_steps': 500,

    'eval_max_episode_steps': 600,
    'eval_num_episodes': 3,
    'eval_render_mode': 'human',
}








