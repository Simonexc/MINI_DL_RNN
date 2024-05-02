import argparse
import os
import yaml
import wandb

from lightning.pytorch.loggers import WandbLogger

from settings import PROJECT, ENTITY, JobType
from trainer.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a wandb experiment using a YAML configuration file."
    )

    # Add an argument for the YAML file path
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML configuration file for the experiment "
        "(assume that parent directory is src/configs/single_runs).",
    )

    # Parse the arguments
    args = parser.parse_args()

    with open(os.path.join("configs", "single_runs", f"{args.yaml_file}.yaml"), "r") as file:
        experiment_config = yaml.safe_load(file)

    with wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type=JobType.TRAINING.value,
        config=experiment_config,
    ) as run:
        config = wandb.config
        wandb_logger = WandbLogger(project=PROJECT, entity=ENTITY)
        data_artifact = run.use_artifact(f"{config.dataset}:latest")
        audio_dir = data_artifact.download()

        config_dict = config.as_dict()

        if hasattr(config, "config_dir"):
            config_artifact = run.use_artifact(
                f"{config.config_dir}:latest")
            config_dict["config_dir"] = config_artifact.download()

        train(config_dict, audio_dir, wandb_logger)
