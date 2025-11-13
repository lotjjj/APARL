import dataclasses
import datetime
import importlib
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Dict, Union, Type, get_origin, get_args
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_path_field(field_type) -> bool:
    """判断字段是否为 Path 类型"""
    return field_type is Path or (hasattr(field_type, '__origin__') and field_type.__origin__ is Path)


def _is_dict_field(field_type) -> bool:
    """判断字段是否为 dict 类型"""
    origin = get_origin(field_type)
    if origin is dict:
        return True
    if isinstance(field_type, type) and issubclass(field_type, dict):
        return True
    return False


def _serialize_value(value: Any, field_type=None) -> Any:
    """递归序列化单个值"""
    if isinstance(value, Path):
        return str(value)
    elif isinstance(value, dict):
        return {k: _serialize_value(v, None) for k, v in value.items()}
    elif hasattr(value, '__dataclass_fields__'):
        # 数据类实例
        return {
            f.name: _serialize_value(getattr(value, f.name), f.type)
            for f in dataclasses.fields(value)
        }
    else:
        return value


def _deserialize_value(value: Any, field_type=None) -> Any:
    """递归反序列化单个值"""
    if _is_path_field(field_type) and isinstance(value, str):
        return Path(value)
    elif _is_dict_field(field_type) and isinstance(value, dict):
        return value  # 字典保持原样，由上层处理
    elif isinstance(value, dict) and field_type and hasattr(field_type, '__dataclass_fields__'):
        # 嵌套数据类
        instance = field_type.__new__(field_type)
        for key, val in value.items():
            if key in field_type.__dataclass_fields__:
                ft = field_type.__dataclass_fields__[key].type
                setattr(instance, key, _deserialize_value(val, ft))
        return instance
    else:
        return value


@dataclass
class BasicConfig:
    # Environment
    env_name: str = 'LunarLander-v3'
    observation_dim: int = field(init=False, default=0)
    action_dim: int = field(init=False, default=0)
    is_discrete: bool = True
    num_envs: int = 6
    vectorization_mode: str = 'async'
    max_episode_steps: int = 300
    options: Dict[str, Any] = field(default_factory=lambda: {})

    # Algorithm
    algorithm: str = None
    is_on_policy: bool = None
    gamma: float = 0.99
    seed: int = 114514

    # Data
    batch_size: int = 64
    horizon_len: int = 1024

    # Model
    policy: str = 'MlpPolicy'
    activation: str = 'relu'
    learning_rate: float = 3e-4
    max_train_steps: int = 100_000_000
    max_grad_norm: float = 2

    # Device
    device: str = 'cpu'

    # Logging
    daytime: str = field(default_factory=lambda: datetime.datetime.now().strftime('%Y%m%d-%H%M'))
    root_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    log_interval: int = 10
    save_interval: int = 1200
    save_dir: Path = field(init=False)
    max_keep: int = 5

    # Evaluation
    eval_num_episodes: int = 2
    eval_max_episode_steps: int = 600
    eval_interval: int = 600
    eval_render_mode: Optional[str] = None
    eval_seed: int = 114514

    def __post_init__(self):
        self._setup_directories()
        self.validate_config()

    def _setup_directories(self):
        self.root_dir = Path.cwd() / 'results' / f'{self.env_name}-{self.daytime}'
        self.config_dir = self.root_dir / 'configs'
        self.log_dir = self.root_dir / 'logs'
        self.save_dir = self.root_dir / 'models'

    def validate_config(self):
        errors = []
        if self.num_envs <= 0: errors.append("num_envs must be > 0")
        if self.batch_size <= 0: errors.append("batch_size must be > 0")
        if self.horizon_len <= 0: errors.append("horizon_len must be > 0")
        if self.max_train_steps <= 0: errors.append("max_train_steps must be > 0")

        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(errors))

    @property
    def hyper_params(self):
        return {
            'algorithm': self.algorithm,
            'is_on_policy': self.is_on_policy,
            'num_envs': self.num_envs,
            'gamma': self.gamma,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'horizon_len': self.horizon_len,
            'max_train_steps': self.max_train_steps,
        }

    def set_env_dim(self, observation_dim: int, action_dim: int):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def print_info(self):
        print(f"==========================================================")
        print(f"   - Algorithm: {self.algorithm}")
        print(f"   - Env: {self.env_name}")
        print(f"   - Device: {self.device}")
        print(f"   - Logs: {self.log_dir}")
        print(f"   - Models: {self.save_dir}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"==========================================================")


@dataclass
class PPOConfig(BasicConfig):
    algorithm: str = 'PPO'
    is_on_policy: bool = True
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    lambda_gae_adv: float = 0.95
    value_coef: float = 0.5
    num_epochs: int = 6
    batch_size: int = 512

    actor_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_dims: List[int] = field(default_factory=lambda: [256, 256])
    actor_lr: float = 2e-5
    critic_lr: float = 3e-5

    def __post_init__(self):
        super().__post_init__()

    @property
    def hyper_params(self):
        orin = super().hyper_params
        orin.update({
            'num_epochs': self.num_epochs,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'actor_dims': self.actor_dims,
            'critic_dims': self.critic_dims,
            'clip_ratio': self.clip_ratio,
            'entropy_coef': self.entropy_coef,
            'lambda_gae_adv': self.lambda_gae_adv,
            'value_coef': self.value_coef,
        })
        return orin

    def validate_config(self):
        super().validate_config()
        errors = []
        total_steps = self.num_envs * self.horizon_len
        if total_steps % self.batch_size != 0:
            errors.append(
                f"Batch_size: {self.batch_size} is not a factor of num_envs*horizon_len: {total_steps}"
            )

        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(errors))

        if not (3 <= self.num_epochs <= 10):
            warnings.warn('num_epochs is not in the range [3,10], which is suggested by PPO paper')


@dataclass
class DQNConfig(BasicConfig):
    algorithm: str = 'DQN'
    is_on_policy: bool = False
    batch_size: int = 128
    target_update_freq: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    def __post_init__(self):
        super().__post_init__()


# =========================
# 配置序列化与反序列化工具
# =========================

def config_to_dict(config: BasicConfig) -> Dict[str, Any]:
    """将配置对象序列化为字典，保留类型信息用于还原"""
    data = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        data[field.name] = _serialize_value(value, field.type)

    # 添加类标识
    data['__config_class__'] = f"{config.__class__.__module__}.{config.__class__.__name__}"
    return data


def save_config(config: BasicConfig, path: Union[str, Path] = None) -> None:
    """保存配置到 YAML 文件"""
    path = Path(path) if path else config.config_dir / f'{config.algorithm}_{config.daytime}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config_to_dict(config)
    with path.open('w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    logger.info(f"Config saved to {path}")


def load_config(path: Union[str, Path]) -> BasicConfig:
    """从 YAML 文件加载配置实例"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data or '__config_class__' not in data:
        raise ValueError("Invalid config file format - missing class identifier")

    class_path = data.pop('__config_class__')

    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load config class '{class_path}': {e}")

    # 创建实例并设置属性
    config = config_class.__new__(config_class)
    for key, value in data.items():
        if key in config_class.__dataclass_fields__:
            field_type = config_class.__dataclass_fields__[key].type
            setattr(config, key, _deserialize_value(value, field_type))

    # 调用 __post_init__
    if hasattr(config, '__post_init__'):
        config.__post_init__()

    logger.info(f"Config loaded from {path} as {config_class.__name__}")
    return config


def mkdir_from_cfg(cfg: BasicConfig):
    """确保所有日志/模型/配置目录存在"""
    for d in [cfg.log_dir, cfg.save_dir, cfg.config_dir]:
        d.mkdir(parents=True, exist_ok=True)


def wrap_config_from_dict(config: BasicConfig, update_dict: Dict[str, Any]) -> BasicConfig:
    """
    根据字典更新配置实例，仅更新存在的字段，自动处理类型转换
    """
    for key, value in update_dict.items():
        if not hasattr(config, key):
            logger.warning(f"Skipping unknown config field: {key}")
            continue

        field_type = config.__dataclass_fields__.get(key).type
        setattr(config, key, _deserialize_value(value, field_type))

    # 重新验证和初始化
    if hasattr(config, 'validate_config'):
        config.validate_config()
    if hasattr(config, '__post_init__'):
        config.__post_init__()

    logger.info("Config updated from dictionary")
    return config