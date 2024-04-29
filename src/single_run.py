import argparse
import os
import yaml

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb

from dataset.training_dataset import SpeechDataset
import models
from models.lightning_model import LightningModel
import dataset.feature_processors as feature_processors
from settings import PROJECT, ENTITY, JobType


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a wandb experiment using a YAML configuration file."
    )

    # Add an argument for the YAML file path
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML configuration file for the experiment "
        "(assume that parent directory is configs).",
    )

    # Parse the arguments
    args = parser.parse_args()

    with open(
        os.path.join("configs", f"{args.yaml_file}.yaml"), "r"
    ) as file:
        experiment_config = yaml.safe_load(file)

    with wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type=JobType.TRAINING.value,
        config=experiment_config,
    ) as run:
        config = wandb.config
        data_artifact = run.use_artifact(f"{config.dataset}:latest")
        audio_dir = data_artifact.download()

        feature_processor: str | None = getattr(config, "feature_processor", None)
        data = SpeechDataset(
            audio_dir,
            config.batch_size,
            feature_processor and getattr(feature_processors, feature_processor)(),
        )
        data.setup()
        wandb_logger = WandbLogger(project=PROJECT, entity=ENTITY)

        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            max_epochs=config.epochs,
        )

        model = getattr(models, config.model_class)(
            **{
                key: getattr(config, key)
                for key in config.model_params
            }
        )
        pl_model = LightningModel(
            model,
            config.model_name,
            config.lr,
        )
        trainer.fit(pl_model, data)
        pl_model.load_best_model()
        trainer.test(pl_model, data)
