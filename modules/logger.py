from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, config):
        self.config = config
        self.tensorboard = SummaryWriter(log_dir=self.config.log_dir, filename_suffix=self.config.algorithm)

    def add_scalar(self, tag: str, value: float, step: int):
        self.tensorboard.add_scalar(tag, value, step)

    def close(self):
        self.tensorboard.close()

