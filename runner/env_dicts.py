

from modules.config import wrap_config_from_dict

LunarLander_v3_dict = {
    'env_name': 'LunarLander-v3',
    'is_discrete': True,

    'max_episode_steps': 300
}

MountainCarContinuous_v0_dict = {
    'env_name': 'MountainCarContinuous-v0',
    'is_discrete': False,

    'device': 'cpu',

    'num_envs': 6,
    'batch_size': 128,

    'eval_render_mode': 'None',
}

BipedalWalker_v3_dict = {
    'env_name': 'BipedalWalker-v3',
    'is_discrete': False,
}









