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

def _make_env(cfg, render_mode: str = None, wrappers=None):
    def _init():
        env = gym.make(cfg.env_name, render_mode=render_mode, max_episode_steps=cfg.max_episode_steps)
        assert cfg.is_discrete == isinstance(env.action_space, gym.spaces.Discrete)
        all_wrappers = wrappers if wrappers else []
        return functools.reduce(lambda e, w: w(e), all_wrappers, env)
    return _init

def make_eval_env(cfg):
    env = gym.make(cfg.env_name, render_mode=cfg.eval_render_mode, max_episode_steps=cfg.eval_max_episode_steps)
    return env

def make_vec_env(cfg, render_mode=None, wrappers=None):
    if cfg.vectorization_mode == 'async':
        envs = AsyncVectorEnv([_make_env(cfg, render_mode, wrappers) for _ in range(cfg.num_envs)])
    elif cfg.vectorization_mode == 'sync':
        envs = SyncVectorEnv([_make_env(cfg, render_mode, wrappers) for _ in range(cfg.num_envs)])
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

def train_agent(envs, eval_env, cfg, model_path: Path =None):
    # agent
    agent = build_agent(cfg)

    if model_path:
        agent.load_model(model_path)

    # info
    cfg.print_info()
    mkdir_from_cfg(cfg)
    save_config(cfg)

    # train
    start_steps = agent.steps
    if start_steps >= cfg.max_train_steps:
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

    with tqdm(total=cfg.max_train_steps, desc='Training ') as pbar:
        onestep = cfg.horizon_len*cfg.num_envs
        pbar.update(start_steps)
        while pbar.n < pbar.total:

            buffer_items = agent.explore(envs) # on_policy: torch.Tensor, off_policy: ndarray

            if cfg.is_on_policy:
                buffer = buffer_items # buffer: Tuple[torch.Tensor, ...]
            else:
                buffer.update_buffer_horizon(buffer_items) # ReplayBuffer: np.ndarray

            agent.update(buffer)
            pbar.update(onestep)

            idx = pbar.n-start_steps

            if idx % (cfg.eval_interval*onestep) == 0:
                evaluate_agent(agent, eval_env, cfg.eval_num_episodes, cfg.eval_seed)

            if idx % (cfg.save_interval*onestep) == 0:
                agent.save_model()

    eval_env.close()
    envs.close()
    agent.save_model()
    pbar.close()
    tqdm.write('Training finished')

def evaluate_agent(agent, env,  test_num, eval_seed):
    # Performance: with no grad
    logger = agent.logger
    with torch.no_grad():
        episode_reward = np.empty(test_num)
        episode_steps = np.zeros(test_num)
        for episode in range(test_num):
            observation, info = env.reset(seed=eval_seed+episode)
            total_reward = 0.0
            steps = 0
            while True:
                observation = torch.from_numpy(observation).to(agent.device)
                action, _ = agent.get_action(observation, deterministic=True)
                np_action = action.cpu().numpy()
                observation, reward, termination, truncation, info = env.step(np_action)
                total_reward += reward
                steps += 1
                if termination or truncation:
                    break
            episode_reward[episode] = total_reward
            episode_steps[episode] = steps
        logger.add_scalar('eval/mean_reward', episode_reward.mean(), agent.steps)
        logger.add_scalar('eval/std_reward', episode_reward.std(), agent.steps)
        logger.add_scalar('eval/max_reward', episode_reward.max(), agent.steps)
        logger.add_scalar('eval/min_reward', episode_reward.min(), agent.steps)
        logger.add_scalar('eval/mean_steps', episode_steps.mean(), agent.steps)



