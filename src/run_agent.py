import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger

from settings import PROJECT, ENTITY
from trainer.train import train


def train_wrapper():
    wandb_logger = WandbLogger(project=PROJECT)
    config = wandb_logger.experiment.config
    config.update(allow_val_change=True)

    data_artifact = wandb_logger.use_artifact(f"{config.dataset}:latest")
    audio_dir = data_artifact.download()

    if hasattr(config, "config_dir"):
        config_artifact = wandb_logger.use_artifact(
            f"{config.config_dir}:latest")
        config.config_dir = config_artifact.download()

    train(config, audio_dir, wandb_logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a wandb agent to execute an experiment."
    )

    parser.add_argument(
        "sweep_id",
        type=str,
        help="Sweep ID provided by sweep.py",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of experiments to perform. Can be None to run indefinitely."
    )

    # Parse the arguments
    args = parser.parse_args()

    wandb.agent(
        args.sweep_id, train_wrapper, count=args.count, project=PROJECT, entity=ENTITY
    )
