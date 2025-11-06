from runner.run import train_agent

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # cfg = load_config(Path.cwd().parent / 'results' / 'LunarLander-v3-20251106' / 'configs' / 'PPO_20251106.yaml')
    # cfg.num_epochs = 7
    # function below suggested
    # wrap_config_from_dict(cfg)
    # model_path = Path.cwd().parent / 'results' / 'LunarLander-v3-20251106' / 'models' / 'PPO_epochs_180.pth'
    # train_agent(cfg,model_path)

    train_agent()

