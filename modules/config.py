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
    num_envs: int = 3
    is_discrete: bool = True
    max_episode_steps: int = 300
    vectorization_mode: str = 'sync'

    # Train
    algorithm: str = field(init=False)
    gamma: float = 0.99
    seed: int = 114514

    # data
    batch_size: int = 511
    horizon_len: int = 1000
    buffer_size: int = 1_000_000

    # model
    policy: str = 'MlpPolicy'
    learning_rate: float = 3e-4
    max_train_steps: int = 100
    max_grad_norm: float = 0.5

    # cuda > intel.xpu > cpu
    device: torch.device =  torch.device(
        'cuda' if torch.cuda.is_available() else (
            'xpu' if torch.xpu.is_available() else
            'cpu')
    )

    # Log
    daytime: str = datetime.datetime.now().strftime('%Y%m%d')
    root_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    log_interval: int = 100
    save_interval: int = 1000
    save_dir: Path = field(init=False)


    def __post_init__(self):

        # 仅可后初始化不会被覆盖的settings

        # Initialize paths after all other fields are set
        self.root_dir = Path.cwd().parent.parent / 'results' / f'{self.env_name}-{self.daytime}'
        self.log_dir = self.root_dir / 'logs'
        self.save_dir = self.root_dir / 'saved_models'
        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def set_env_dim(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    @property
    def count_saved(self):
        assert len(list(self.save_dir.glob('*.pth'))) == len(list(self.log_dir.glob('*.log')))
        return len(list(self.save_dir.glob('*.pth')))

    @property
    def log_path(self):
        return self.log_dir / f"{self.algorithm}-#{self.count_saved}-{datetime.datetime.now().strftime('%H:%M')}.log"

    @property
    def save_path(self):
        return self.save_dir / f"{self.algorithm}-#{self.count_saved}-{datetime.datetime.now().strftime('%H:%M')}.pth"


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
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4


    def __post_init__(self):

        self.distribution = torch.distributions.Normal if not self.is_discrete else torch.distributions.Categorical
        # Initialize lists after other fields are set
        self.actor_dims = [256, 256, 256]
        self.critic_dims = [256, 256, 256]

        super().__post_init__()

@dataclass
class DQNConfig(BasicConfig):
    pass


# test
if __name__ == '__main__':
    config = tyro.cli(PPOConfig)
    print(config.log_path)
    print(config.save_path)
    print(config.count_saved)