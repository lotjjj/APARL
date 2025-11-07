import dataclasses
import datetime
import importlib
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BasicConfig:
    # Environment
    env_name: str = 'LunarLander-v3'
    observation_dim: int = field(init=False, default=0)
    action_dim: int = field(init=False, default=0)
    is_discrete: bool = True
    num_envs: int = 6
    vectorization_mode: str = 'async'
    options: Dict[str, Any] = field(default_factory=lambda: {})

    # Algorithm
    algorithm: str = None
    gamma: float = 0.99
    seed: int = 114514
    num_epochs: int = 3

    # Data
    batch_size: int = 64
    horizon_len: int = 300
    buffer_size: int = 1_000_000

    # Model
    policy: str = 'MlpPolicy'
    learning_rate: float = 3e-4
    max_train_epochs: int = 1_000_000
    max_grad_norm: float = 0.5

    # Device
    device: str = 'cpu'

    # Logging
    daytime: str = field(default_factory=lambda: datetime.datetime.now().strftime('%Y%m%d-%H%M'))
    root_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    log_interval: int = 10
    save_interval: int = 600
    save_dir: Path = field(init=False)
    max_keep: int = 5

    # Evaluation
    eval_num_episodes: int = 10
    eval_max_episode_steps: int = 1000
    eval_interval: int = 200
    eval_render_mode: Optional[str] = None

    def __post_init__(self):
        self._setup_directories()
        self.validate_config()

    def _setup_directories(self):
        self.root_dir = Path.cwd().parent / 'results' / f'{self.env_name}-{self.daytime}'
        self.config_dir = self.root_dir / 'configs'
        self.log_dir = self.root_dir / 'logs'
        self.save_dir = self.root_dir / 'models'

    def validate_config(self):
        errors = []
        if self.num_envs <= 0: errors.append("num_envs must be > 0")
        if self.batch_size <= 0: errors.append("batch_size must be > 0")
        if self.horizon_len <= 0: errors.append("horizon_len must be > 0")
        if self.max_train_epochs <= 0: errors.append("max_train_epochs must be > 0")
        if self.num_envs * self.horizon_len < self.batch_size * 2:
            errors.append("num_envs * horizon_len should >= batch_size * 2")

        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(errors))

    def set_env_dim(self, observation_dim: int, action_dim: int):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def print_info(self):
        print(f"   - Algorithm: {self.algorithm}")
        print(f"   - Env: {self.env_name}")
        print(f"   - Device: {self.device}")
        print(f"   - Logs: {self.log_dir}")
        print(f"   - Models: {self.save_dir}")


@dataclass
class PPOConfig(BasicConfig):
    algorithm: str = 'PPO'
    is_on_policy: bool = True
    clip_ratio: float = 0.2
    entropy_coef: float = 0.1
    lambda_gae_adv: float = 0.95
    value_coef: float = 0.5
    num_epochs: int = 3
    batch_size: int = 511

    actor_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    critic_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    actor_lr: float = 2e-5
    critic_lr: float = 3e-4

    def __post_init__(self):
        super().__post_init__()


@dataclass
class DQNConfig(BasicConfig):
    algorithm: str = 'DQN'
    target_update_freq: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    def __post_init__(self):
        super().__post_init__()

# Qwen3 Contributions

def save_config(config: BasicConfig, path: Union[str, Path] = None) -> None:

    if path:
        path = Path(path)
    else:
        path = config.config_dir / f'{config.algorithm}_{config.daytime}.yaml'

    # 准备可序列化的字典
    data = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)

        # 特殊处理Path对象
        if isinstance(value, Path):
            value = str(value)

        # 特殊处理字典对象（包括options和其他Dict字段）
        elif isinstance(value, dict):
            # 递归处理字典中的Path对象
            value = _serialize_dict(value)

        # 递归处理嵌套配置对象（如果未来有）
        elif hasattr(value, '__dataclass_fields__'):
            value = {k: str(v) if isinstance(v, Path) else _serialize_dict(v) if isinstance(v, dict) else v
                     for k, v in value.__dict__.items()}

        data[field.name] = value

    # 添加类标识信息用于类型还原
    data['__config_class__'] = (
        f"{config.__class__.__module__}.{config.__class__.__name__}"
    )

    # 保存到YAML
    with path.open('w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    logger.info(f"Config saved to {path}")

def _serialize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """递归序列化字典，处理嵌套的字典和Path对象"""
    result = {}
    for k, v in d.items():
        if isinstance(v, Path):
            result[k] = str(v)
        elif isinstance(v, dict):
            result[k] = _serialize_dict(v)
        elif hasattr(v, '__dataclass_fields__'):
            # 如果字典值是数据类实例，也进行序列化
            nested_dict = {}
            for field_name in dataclasses.fields(v):
                field_value = getattr(v, field_name.name)
                if isinstance(field_value, Path):
                    nested_dict[field_name.name] = str(field_value)
                elif isinstance(field_value, dict):
                    nested_dict[field_name.name] = _serialize_dict(field_value)
                else:
                    nested_dict[field_name.name] = field_value
            result[k] = nested_dict
        else:
            result[k] = v
    return result


def load_config(path: Union[str, Path]) -> BasicConfig:
    """
    从YAML文件加载配置实例
    自动还原原始配置类类型
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # 加载YAML数据
    with path.open('r') as f:
        data = yaml.safe_load(f)

    if not data or '__config_class__' not in data:
        raise ValueError("Invalid config file format - missing class identifier")

    # 提取类信息并移除特殊键
    class_path = data.pop('__config_class__')

    # 动态导入配置类
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load config class '{class_path}': {e}")

    # 创建实例（绕过__init__，直接设置属性）
    config = config_class.__new__(config_class)

    # 设置所有属性（包括init=False的字段）
    for key, value in data.items():
        if key in config_class.__dataclass_fields__:
            field_type = config_class.__dataclass_fields__[key].type

            # 特殊处理Path字段
            if field_type is Path and isinstance(value, str):
                value = Path(value)

            # 特殊处理字典字段（包括options和其他Dict字段）
            elif (hasattr(field_type, '__origin__') and
                  field_type.__origin__ is dict) or \
                 (isinstance(field_type, type) and issubclass(field_type, dict)):
                value = _deserialize_dict(value)

            # 处理嵌套配置（如果未来有）
            elif isinstance(value, dict) and hasattr(config_class, key):
                nested_config = getattr(config, key)
                if hasattr(nested_config, '__dataclass_fields__'):
                    for nk, nv in value.items():
                        if nk in nested_config.__dataclass_fields__:
                            field_type = nested_config.__dataclass_fields__[nk].type
                            if field_type is Path and isinstance(nv, str):
                                nv = Path(nv)
                            elif (hasattr(field_type, '__origin__') and
                                  field_type.__origin__ is dict) or \
                                 (isinstance(field_type, type) and issubclass(field_type, dict)):
                                nv = _deserialize_dict(nv)
                            setattr(nested_config, nk, nv)
                    continue

            setattr(config, key, value)

    # 确保__post_init__被调用
    if hasattr(config, '__post_init__'):
        config.__post_init__()

    logger.info(f"Config loaded from {path} as {config_class.__name__}")
    return config

def _deserialize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """递归反序列化字典，还原嵌套的字典和Path对象"""
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            # 尝试将字符串转换为Path对象（如果它看起来像路径）
            # 这里我们假设所有字符串都可能是路径，根据实际需求调整
            # 为了更精确，我们只处理以特定模式结尾的字符串
            result[k] = v
        elif isinstance(v, dict):
            result[k] = _deserialize_dict(v)
        else:
            result[k] = v
    return result

def mkdir_from_cfg(cfg: BasicConfig):
    for d in [cfg.log_dir, cfg.save_dir, cfg.config_dir]:
        d.mkdir(parents=True, exist_ok=True)

def wrap_config_from_dict(config: BasicConfig, update_dict: Dict[str, Any]) -> Any:
    """
    根据字典更新配置实例
    只更新配置类中存在的字段，自动处理Path类型转换
    """

    for key, value in update_dict.items():
        if not hasattr(config, key):
            logger.warning(f"Skipping unknown config field: {key}")
            continue

        # 获取字段类型
        field_type = type(getattr(config, key))

        # 自动转换Path类型
        if field_type is Path and isinstance(value, str):
            value = Path(value)
        elif field_type is str and isinstance(value, Path):
            value = str(value)

        # 递归处理嵌套配置（如果未来有）
        if hasattr(value, 'items') and hasattr(config, key):
            nested_config = getattr(config, key)
            if hasattr(nested_config, '__dataclass_fields__'):
                wrap_config_from_dict(nested_config, value)
                continue

        setattr(config, key, value)

    # 重新验证配置
    if hasattr(config, 'validate_config'):
        config.validate_config()

    logger.info("Config updated from dictionary")
    config.__post_init__()
    return config