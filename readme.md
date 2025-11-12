# APA RL

A PyTorch-based reinforcement learning framework that supports PPO and DQN algorithms, featuring modular design and flexible configuration system.

## Project Structure

```
ApaRL/
├── agents/              # Agent implementations
│   ├── AgentBase.py     # Base agent class
│   ├── AgentPPO.py      # PPO agent
│   └── AgentDQN.py      # DQN agent
├── modules/             # Core modules
│   ├── config.py        # Configuration management
│   ├── Extractor.py     # Feature extractor
│   ├── Head.py          # Network heads
│   ├── ReplayBuffer.py  # Experience replay buffer
│   └── logger.py        # Logging system
├── runner/              # Training runner
│   ├── run.py           # Main training logic
│   └── train_dicts.py   # Predefined configuration dictionaries
└── trainer.py           # Training entry script
```

## Quick Start

### 1. Requirements

```bash
pip install torch gymnasium numpy tqdm
```

### 2. Basic Usage

#### Method 1: Using Predefined Configurations

```python
from runner import train_dicts
from modules.config import PPOConfig, wrap_config_from_dict
from runner.run import train_agent, make_vec_env, make_eval_env

# Use predefined LunarLander configuration
cfg = wrap_config_from_dict(PPOConfig(), train_dicts.LunarLander_v3_dict)

# Create training and evaluation environments
envs = make_vec_env(cfg)
eval_env = make_eval_env(cfg)

# Start training
train_agent(envs, eval_env, cfg)
```

#### Method 2: Direct Training Script

```bash
python trainer.py
```

### 3. Configuration System

#### Dictionary Configuration (Recommended)

```python
from modules.config import PPOConfig, wrap_config_from_dict

# Define configuration dictionary
my_config = {
    'env_name': 'CartPole-v1',
    'is_discrete': True,
    'num_envs': 8,
    'horizon_len': 512,
    'batch_size': 256,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'max_train_steps': 1000000,
}

# Create configuration object
cfg = wrap_config_from_dict(PPOConfig(), my_config)
```

#### Class Inheritance Configuration

```python
from modules.config import PPOConfig

@dataclass
class MyPPOConfig(PPOConfig):
    actor_dims: List[int] = field(default_factory=lambda: [512, 256])
    critic_dims: List[int] = field(default_factory=lambda: [512, 256])
    entropy_coef: float = 0.02
```

### 4. Predefined Environment Configurations

#### Discrete Action Space (LunarLander-v3)

```python
from runner.train_dicts import LunarLander_v3_dict
cfg = wrap_config_from_dict(PPOConfig(), LunarLander_v3_dict)
```

#### Continuous Action Space (BipedalWalker-v3)

```python
from runner.train_dicts import BipedalWalker_v3_dict
cfg = wrap_config_from_dict(PPOConfig(), BipedalWalker_v3_dict)
```

## Configuration Parameters

### Basic Configuration (BasicConfig)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env_name` | str | 'LunarLander-v3' | Environment name |
| `num_envs` | int | 6 | Number of parallel environments |
| `vectorization_mode` | str | 'async' | Vectorization mode ('async'/'sync') |
| `max_episode_steps` | int | 300 | Maximum steps per episode |
| `device` | str | 'cpu' | Computing device |
| `max_train_steps` | int | 100_000_000 | Maximum training steps |

### PPO Configuration (PPOConfig)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_ratio` | float | 0.2 | PPO clipping ratio |
| `entropy_coef` | float | 0.01 | Entropy coefficient |
| `lambda_gae_adv` | float | 0.95 | GAE advantage function lambda |
| `value_coef` | float | 0.5 | Value function coefficient |
| `num_epochs` | int | 6 | Number of training epochs per update |
| `actor_dims` | List[int] | [256, 256] | Actor network dimensions |
| `critic_dims` | List[int] | [256, 256] | Critic network dimensions |
| `actor_lr` | float | 2e-5 | Actor learning rate |
| `critic_lr` | float | 3e-5 | Critic learning rate |


## Training Monitoring

The following files are automatically generated during training:

- `swanlog/run-*/`: Training log directory
- `results/{env_name}-{date}/`: Model and configuration save directory
  - `models/`: Saved model files
  - `configs/`: Saved configuration files
  - `logs/`: Training logs

## Example Training Commands

```bash
# Train LunarLander environment
python trainer.py

# Custom configuration training
python -c "
from runner import train_dicts
from modules.config import PPOConfig, wrap_config_from_dict
from runner.run import train_agent, make_vec_env, make_eval_env

cfg = wrap_config_from_dict(PPOConfig(), {
    'env_name': 'CartPole-v1',
    'max_train_steps': 500000,
    'eval_interval': 100
})

envs = make_vec_env(cfg)
eval_env = make_eval_env(cfg)
train_agent(envs, eval_env, cfg)
"
```

## TODO

- [ ] Tensorboard integration
- [ ] Training curve visualization
- [ ] DQN algorithm improvement
- [ ] Learning rate scheduler
- [ ] More predefined environment configurations
- [ ] Model evaluation and inference scripts
- [ ] Distributed training support

---

**Note**: This project is still under active development. Some features may be unstable. It is recommended to conduct thorough testing before using in production environments.