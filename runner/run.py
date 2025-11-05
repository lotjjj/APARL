import torch
import tqdm
import tyro
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from sympy.abc import lamda

from modules.config import PPOConfig

import gymnasium as gym
import numpy as np


def make_env(cfg):
    def _init():
        single_env = gym.make(cfg.env_name, max_episode_steps=cfg.max_episode_steps)
        if not cfg.is_discrete:
            single_env = gym.wrappers.RescaleAction(single_env,
                                                    min_action=-np.ones(shape=single_env.action_space.shape),
                                                    max_action=np.ones(shape=single_env.action_space.shape))
        single_env = gym.wrappers.FlattenObservation(single_env)
        return single_env
    return _init


def make_vec_env(cfg):
    # Make env
    # wrapper
    # read dim save to config
    if cfg.vectorization_mode == 'async':
        envs = AsyncVectorEnv([make_env(cfg) for _ in range(cfg.num_envs)])
    elif cfg.vectorization_mode == 'sync':
        envs = SyncVectorEnv([make_env(cfg) for _ in range(cfg.num_envs)])
    else:
        raise ValueError(f'Unsupported vectorization mode: {cfg.vectorization_mode}')

    # Determine action space dimension based on whether it's discrete or continuous
    if cfg.is_discrete:
        action_dim = envs.single_action_space.n
    else:
        action_dim = envs.single_action_space.shape[0]

    cfg.set_env_dim(envs.single_observation_space.shape[0], action_dim)
    return envs

def build_logger(cfg):
    return None

def build_agent(cfg):
    agent = None
    try:
        alg = cfg.algorithm.lower()
        if alg== 'ppo':
            from agents.AgentPPO import AgentPPO
            agent =  AgentPPO(cfg)
        elif alg == 'dqn':
            from agents.AgentDQN import AgentDQN

        else:
            raise ValueError(f'Unsupported algorithm: {cfg.algorithm}')

    except Exception as e:
        print(f'Error: {e}')
        print('Try to build environment first')

    print(f'config -> {cfg.algorithm} agent and {cfg.env_name} env have been successfully built')
    return agent


def train_agent(cfg):
    # multiprocessing
    # build env
    # build agent
    # build logger

    envs = make_vec_env(cfg)
    agent = build_agent(cfg)

    observation, info = envs.reset()
    if cfg.num_envs == 1:
        assert observation.shape == (cfg.observation_dim,)
        assert isinstance(observation, np.ndarray)
        observation = torch.from_numpy(observation).to(cfg.device).unsqueeze(0)
    else:
        observation = torch.from_numpy(observation).to(cfg.device)
    assert  observation.shape == (cfg.num_envs, cfg.observation_dim)
    assert isinstance(observation, torch.Tensor)

    agent.last_observation =  observation.detach()

    pbar = tqdm.tqdm(range(cfg.max_train_steps))

    for _ in pbar:

        buffer = agent.explore(cfg.horizon_len)

        actor_loss, critic_loss = agent.update(buffer)

        pbar.set_postfix(actor_loss=actor_loss, critic_loss=critic_loss)

    envs.close()


if __name__ == '__main__':

    cfg = tyro.cli(PPOConfig)

    if cfg.vectorization_mode == 'async':
        envs = AsyncVectorEnv([make_env(cfg) for _ in range(cfg.num_envs)])
    elif cfg.vectorization_mode == 'sync':
        envs = SyncVectorEnv([make_env(cfg) for _ in range(cfg.num_envs)])
    else:
        raise ValueError(f'Unsupported vectorization mode: {cfg.vectorization_mode}')

    # Determine action space dimension based on whether it's discrete or continuous
    if cfg.is_discrete:
        action_dim = envs.single_action_space.n
    else:
        action_dim = envs.single_action_space.shape[0]

    cfg.set_env_dim(envs.single_observation_space.shape[0], action_dim)

    agent = build_agent(cfg)

    observation, info = envs.reset()
    if cfg.num_envs == 1:
        assert observation.shape == (cfg.observation_dim,)
        assert isinstance(observation, np.ndarray)
        observation = torch.from_numpy(observation).to(cfg.device).unsqueeze(0)
    else:
        observation = torch.from_numpy(observation).to(cfg.device)
    assert observation.shape == (cfg.num_envs, cfg.observation_dim)
    assert isinstance(observation, torch.Tensor)

    agent.last_observation = observation.detach()

    pbar = tqdm.tqdm(range(cfg.max_train_steps))

    for _ in pbar:
        buffer = agent.explore(envs)

        actor_loss, critic_loss = agent.update(buffer)

        pbar.set_postfix(actor_loss=actor_loss, critic_loss=critic_loss)

    envs.close()





