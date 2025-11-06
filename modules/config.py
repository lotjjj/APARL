import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch
import tyro

@dataclass
class BasicConfig:
    # Env
    env_name: str = 'LunarLander-v3'
    observation_dim: int = field(init=False)
    action_dim: int = field(init=False)
    is_discrete: bool = True
    num_envs: int = 4
    max_episode_steps: int = 300
    vectorization_mode: str = 'async'

    # Train
    algorithm: str = field(init=False)
    gamma: float = 0.99
    seed: int = 114514
    num_epochs: int = 6

    # data
    batch_size: int = 511
    horizon_len: int = 600
    buffer_size: int = 1_000_000

    # model
    policy: str = 'MlpPolicy'
    learning_rate: float = 3e-4
    max_train_epochs: int = 100_000
    max_grad_norm: float = 0.5

    # cuda > intel.xpu > cpu
    device: str = 'cpu'

    # Log
    daytime: str = datetime.datetime.now().strftime('%Y%m%d')
    root_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    log_interval: int = 5
    save_interval: int = 30
    save_dir: Path = field(init=False)
    max_keep: int = 5

    # eval
    eval_num_episodes: int = 5
    eval_max_episode_steps: int = 500
    eval_interval: int = 10
    eval_render_mode: str = None


    def __post_init__(self):

        # Initialize paths after all other fields are set
        self.root_dir = Path.cwd().parent / 'results' / f'{self.env_name}-{self.daytime}'
        self.log_dir = self.root_dir / 'logs'
        self.save_dir = self.root_dir / 'saved_models'
        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = self.num_epochs * self.log_interval
        self.save_interval = self.num_epochs * self.save_interval
        self.eval_interval = self.num_epochs * self.eval_interval

        # Validation
        assert self.num_envs * self.horizon_len >= self.batch_size * 2

    def set_env_dim(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def _print_device(self):
        print(f'Using device: {self.device}')

    def _print_dir(self):
        print(f'Root dir: {self.root_dir}')

    def print_info(self):
        self._print_device()
        self._print_dir()

@dataclass
class PPOConfig(BasicConfig):
    algorithm: str = 'PPO'

    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    lambda_gae_adv: float = 0.95
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 6
    batch_size: int = 511

    actor_dims: List[int] = field(init=False)
    critic_dims: List[int] = field(init=False)
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4

    def __post_init__(self):
        # Initialize lists after other fields are set
        self.actor_dims = [128, 128, 128]
        self.critic_dims = [128, 128, 128]
        super().__post_init__()


@dataclass
class DQNConfig(BasicConfig):
    pass