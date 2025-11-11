
LunarLander_v3_dict = {
    'env_name': 'LunarLander-v3',
    'is_discrete': True,
    'device': 'cpu',

    'horizon_len': 2048,
    'num_envs': 6,
    'batch_size': 256,
    'entropy_coef': 0.01,
    'clip_ratio': 0.05,
    'lambda_gae_adv': 0.95,
    'num_epochs': 4,
    'max_grad_norm': 1.0,

    'max_episode_steps': 800,
    'eval_max_episode_steps': 400,
    'eval_num_episodes': 1,
    'eval_render_mode': None,
    'eval_seed': 42,

    'actor_dims': [256, 256, 256],
    'critic_dims': [256, 256, 256],
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,

    'eval_interval': 5,
    'save_interval': 500,
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








