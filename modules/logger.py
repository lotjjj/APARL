from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import swanlab
from modules.config import  BasicConfig


class Logger:
    def __init__(self, config: BasicConfig):

        swanlab.init(
            project= f"{config.env_name}-{'Discrete' if config.is_discrete else 'Continuous'}",
            experiment_name=f"BasicAlgorithmTest-{config.algorithm}",

            description="Test our implementation",

            tags=[
                "https://github.com/lotjjj/APARL.git",
            ],

            config = config.hyper_params
        )
        # hyperparams

        self.config = config
        swanlab.sync_tensorboard_torch()
        self.tensorboard = SummaryWriter(log_dir=str(self.config.log_dir), filename_suffix=self.config.algorithm)
        self.pbar = None

    def add_scalar(self, tag: str, value: float, step: int):
        self.tensorboard.add_scalar(tag, value, step)

    def close(self):
        self.tensorboard.close()
        self.pbar.close()

    def setup_pbar(self, total_steps: int = None):
        if total_steps is None:
            total_steps = self.config.max_train_steps
        self.pbar = tqdm(total=total_steps, desc=f'Training, {self.config.algorithm} ')
