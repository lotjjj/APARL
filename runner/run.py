from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from modules.ReplayBuffer import ReplayBuffer
from modules.config import save_config, mkdir_from_cfg

import gymnasium as gym
import numpy as np
import functools


def make_env(cfg, render_mode: str = None, wrappers=None):
    def _init():
        env = gym.make(cfg.env_name, render_mode=render_mode, max_episode_steps=cfg.max_episode_steps)
        assert cfg.is_discrete == isinstance(env.action_space, gym.spaces.Discrete)
        all_wrappers = wrappers if wrappers else []
        return functools.reduce(lambda e, w: w(e), all_wrappers, env)
    return _init


def make_vec_env(cfg, render_mode=None, wrappers=None):
    if cfg.vectorization_mode == 'async':
        envs = AsyncVectorEnv([make_env(cfg, render_mode, wrappers) for _ in range(cfg.num_envs)])
    elif cfg.vectorization_mode == 'sync':
        envs = SyncVectorEnv([make_env(cfg, render_mode, wrappers) for _ in range(cfg.num_envs)])
    else:
        raise ValueError(f'Unsupported vectorization mode: {cfg.vectorization_mode}')

    action_dim = envs.single_action_space.n if cfg.is_discrete else envs.single_action_space.shape[0]
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
    return agent

def train_agent(envs, cfg, model_path: Path =None):
    # agent
    agent = build_agent(cfg)

    if model_path:
        agent.load_model(model_path)

    # info
    cfg.print_info()
    mkdir_from_cfg(cfg)
    save_config(cfg)

    # train
    start_epoch = agent.epochs
    if start_epoch >= cfg.max_train_epochs:
        print(f'Training has already been completed, no need to train again')
        return

    # init
    buffer = () if cfg.is_on_policy else ReplayBuffer(cfg)

    observation, info = envs.reset(options=cfg.options)

    if cfg.num_envs == 1:
        assert observation.shape == (cfg.observation_dim,)
        assert isinstance(observation, np.ndarray)
        observation = torch.from_numpy(observation).to(cfg.device).unsqueeze(0)
    else:
        observation = torch.from_numpy(observation).to(cfg.device)
    assert  observation.shape == (cfg.num_envs, cfg.observation_dim)
    assert isinstance(observation, torch.Tensor)

    agent.last_observation =  observation.detach()

    with tqdm(total=cfg.max_train_epochs, desc='Training ') as pbar:
        pbar.update(start_epoch)
        while pbar.n < pbar.total:

            buffer_items = agent.explore(envs) # on_policy: torch.Tensor, off_policy: ndarray

            if cfg.is_on_policy:
                buffer = buffer_items # buffer: Tuple[torch.Tensor, ...]
            else:
                buffer.update_buffer_horizon(buffer_items) # ReplayBuffer: np.ndarray

            agent.update(buffer)
            pbar.update(cfg.num_epochs)

            idx = pbar.n-start_epoch

            if idx % (cfg.eval_interval*cfg.num_epochs) == 0:
                mean, std, seq, steps  = evaluate_agent(agent,cfg)
                tqdm.write(f'\nEvaluate agent at epoch {pbar.n}, eval episodes: {cfg.eval_num_episodes}, '
                           f'mean reward: {mean:.4f}, std: {std:.4f}, max_reward: {seq.max():.4f}, min_reward: {seq.min():.4f}, '
                           f'steps: {steps.mean()}/{cfg.eval_max_episode_steps}')

            if idx % (cfg.save_interval*cfg.num_epochs) == 0:
                agent.save_model(pbar.n)

    envs.close()
    agent.save_model(pbar.n)
    pbar.close()
    tqdm.write('Training finished')

def evaluate_agent(agent, cfg):
    env = make_env(cfg, render_mode=cfg.eval_render_mode)()

    # Performance: with no grad
    with torch.no_grad():
        episode_reward = np.empty(cfg.eval_num_episodes)
        episode_steps = np.zeros(cfg.eval_num_episodes)
        for episode in range(cfg.eval_num_episodes):
            observation, info = env.reset(seed=cfg.eval_seed+episode)
            total_reward = 0.0
            steps = 0
            while True:
                observation = torch.from_numpy(observation).to(cfg.device)
                action, _ = agent.get_action(observation, deterministic=True)
                np_action = action.cpu().numpy()
                observation, reward, termination, truncation, info = env.step(np_action)
                total_reward += reward
                steps += 1
                if termination or truncation or steps >= cfg.eval_max_episode_steps:
                    break
            episode_reward[episode] = total_reward
            episode_steps[episode] = steps
        mean_reward, std_reward = episode_reward.mean(), episode_reward.std()

        env.close()

        return mean_reward, std_reward, episode_reward, episode_steps




