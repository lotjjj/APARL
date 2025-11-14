# Discrete
LunarLander_v3_dict = {
    'env_name': 'LunarLander-v3',
    'is_discrete': True,
    'device': 'cpu',

    'horizon_len': 512,
    'num_envs': 8,
    'batch_size': 256,
    'entropy_coef': 0.01,
    'clip_ratio': 0.2,
    'lambda_gae_adv': 0.95,
    'num_epochs': 10,
    'max_grad_norm': 0.5,

    'max_episode_steps': 200,
    'eval_max_episode_steps': 200,
    'eval_num_episodes': 20,
    'eval_render_mode': None,
    'eval_seed': 114514,

    'actor_dims': [256, 256],
    'critic_dims': [256, 256],
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,

    'eval_interval': 10,
    'save_interval': 200,
}

# Continuous
BipedalWalker_v3_dict = {
    'env_name': 'BipedalWalker-v3',
    'is_discrete': False,
    'device': 'cpu',
    'activation': 'relu',

    'horizon_len': 1024,
    'num_envs': 16,
    'batch_size': 1024,
    'entropy_coef': 0.001,
    'clip_ratio': 0.2,
    'value_coef': 0.5,
    'lambda_gae_adv': 0.95,
    'num_epochs': 8,
    'max_grad_norm': 0.5,
    'seed': 42,

    'max_episode_steps': 999,
    'eval_max_episode_steps': 999,
    'eval_num_episodes': 5,
    'eval_render_mode': None,
    'eval_seed': 114514,

    'actor_dims': [64, 128, 64],
    'critic_dims': [64, 128, 64],
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,

    'eval_interval': 10,
    'save_interval': 200,
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







