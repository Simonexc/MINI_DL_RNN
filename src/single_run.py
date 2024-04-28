import argparse
import os
import yaml

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb

from dataset.training_dataset import SpeechDataset
from models.transformer import TransformerModel
from settings import PROJECT, ENTITY, JobType, ALL_CLASSES


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
            project=PROJECT, entity=ENTITY, job_type=JobType.TRAINING.value, config=experiment_config
    ) as run:
        config = wandb.config
        data_artifact = run.use_artifact(f"{config.dataset}:latest")
        audio_dir = data_artifact.download()
        data = SpeechDataset(audio_dir, config.batch_size)
        data.setup()
        wandb_logger = WandbLogger(project=PROJECT)

        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            max_epochs=config.epochs,
        )

        model = TransformerModel(
            config.lr,
            int(16000 * config.time_interval_grouping),
            config.embedding_hidden_layer,
            config.hidden_vector_size,
            config.heads_num,
            config.hidden_layer,
            config.layers,
            len(ALL_CLASSES),
        )
        trainer.fit(model, data)
        trainer.test(model, data)
