from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import swanlab
from modules.config import config_to_dict

class Logger:
    def __init__(self, config):

        swanlab.init(
            project="LunarLander-Discrete",
            experiment_name="BasicAlgorithmTest",

            description="Test our implementation",

            tags=[
                "https://github.com/lotjjj/APARL.git",
            ],

            config = config.hyper_params
        )
        # hyperparams

        self.config = config
        swanlab.sync_tensorboard_torch()
        self.tensorboard = SummaryWriter(log_dir=self.config.log_dir, filename_suffix=self.config.algorithm)
        self.pbar = tqdm(total=self.config.max_train_steps, desc='Training ')

    def add_scalar(self, tag: str, value: float, step: int):
        self.tensorboard.add_scalar(tag, value, step)

    def close(self):
        self.tensorboard.close()
        self.pbar.close()

