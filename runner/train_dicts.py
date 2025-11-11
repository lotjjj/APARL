
LunarLander_v3_dict = {
    'env_name': 'LunarLander-v3',
    'is_discrete': True,
    'device': 'cpu',

    'horizon_len': 2400,
    'num_envs': 4,
    'batch_size': 511,
    'entropy_coef': 0.05,
    'clip_ratio': 0.1,
    'lambda_gae_adv': 0.95,
    'num_epochs': 8,
    'max_grad_norm': 0.5,

    'max_episode_steps': 800,
    'eval_max_episode_steps': 800,
    'eval_num_episodes': 1,
    'eval_render_mode': None,

    'actor_dims': [256, 256, 256],
    'critic_dims': [256, 256, 256],
    'actor_lr': 2e-4,
    'critic_lr': 2e-4,

    'eval_interval': 10,
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








