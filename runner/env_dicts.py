

from modules.config import wrap_config_from_dict

LunarLander_v3_dict = {
    'env_name': 'LunarLander-v3',
    'is_discrete': True,

    'max_episode_steps': 300
}

MountainCar_v0_dict = {
    'env_name': 'MountainCarContinuous-v0',
    'is_discrete': False,

    'num_envs': 12,
    'batch_size': 1023,
}

BipedalWalker_v3_dict = {
    'env_name': 'BipedalWalker-v3',
    'is_discrete': False,
}









