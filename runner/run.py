import torch
import tqdm
import tyro
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from modules.config import PPOConfig

import gymnasium as gym
import numpy as np


def make_env(cfg, render_mode: str = None):
    def _init():
        single_env = gym.make(cfg.env_name, max_episode_steps=cfg.max_episode_steps, render_mode= render_mode)
        if not cfg.is_discrete:
            single_env = gym.wrappers.RescaleAction(single_env,
                                                    min_action=-np.ones(shape=single_env.action_space.shape),
                                                    max_action=np.ones(shape=single_env.action_space.shape))
        single_env = gym.wrappers.FlattenObservation(single_env)
        return single_env
    return _init


def make_vec_env(cfg, render_mode=None):
    if cfg.vectorization_mode == 'async':
        envs = AsyncVectorEnv([make_env(cfg, render_mode=render_mode) for _ in range(cfg.num_envs)])
    elif cfg.vectorization_mode == 'sync':
        envs = SyncVectorEnv([make_env(cfg, render_mode=render_mode) for _ in range(cfg.num_envs)])
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

    print(f'config -> {cfg.algorithm} agent and {cfg.env_name} env have been successfully built')
    return agent

def train_agent(cfg):
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

        buffer = agent.explore(envs)

        actor_loss, critic_loss, entropy_loss = agent.update(buffer)

        pbar.set_postfix(actor_loss=actor_loss, critic_loss=critic_loss, entropy_loss=entropy_loss)

        if _ % cfg.save_interval == 0 and _ != 0:
            agent.save_model()

        if _ % cfg.eval_interval == 0:
            mean, std, seq, steps  = evaluate_agent(agent,cfg)
            print(f'\nEvaluating agent at step {_}, eval episodes: {cfg.eval_num_episodes}, mean reward: {mean:.2f}, std: {std:.2f}, steps: {steps.mean()}')

    envs.close()


def evaluate_agent(agent, cfg):
    env = gym.make(cfg.env_name, max_episode_steps=cfg.eval_max_episode_steps, render_mode= cfg.eval_render_mode)
    with torch.no_grad():
        episode_reward = np.empty(cfg.eval_num_episodes)
        episode_steps = np.zeros(cfg.eval_num_episodes)
        for episode in range(cfg.eval_num_episodes):
            observation, info = env.reset()
            total_reward = 0.0
            steps = 0
            while True:
                observation = torch.from_numpy(observation).to(cfg.device)
                action, _ = agent.get_action(observation)
                np_action = action.detach().cpu().numpy()
                observation, reward, termination, truncation, info = env.step(np_action)
                total_reward += reward
                steps += 1
                if termination or truncation:
                    break
            episode_reward[episode] = total_reward
            episode_steps[episode] = steps
        mean_reward, std_reward = episode_reward.mean(), episode_reward.std()
        env.close()
        return mean_reward, std_reward, episode_reward, episode_steps

if __name__ == '__main__':
    cfg = tyro.cli(PPOConfig)
    print(f'Using device: {cfg.device}')
    train_agent(cfg)




