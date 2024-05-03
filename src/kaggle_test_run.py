import argparse
import os
import glob
import wandb

from lightning.pytorch.loggers import WandbLogger
import pandas as pd

from settings import PROJECT, ENTITY, JobType, ArtifactType
from trainer.test import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Kaggle test."
    )

    # Add an argument for the YAML file path
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the artifact with model that you want to test.",
    )

    # Parse the arguments
    args = parser.parse_args()

    experiment_config = {
        "model_name": args.model_name,
    }

    with wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type=JobType.KAGGLE_TEST.value,
        config=experiment_config,
    ) as run:
        config = wandb.config
        wandb_logger = WandbLogger(project=PROJECT, entity=ENTITY)
        data_artifact = run.use_artifact(f"speech-test-waveform:latest")
        audio_dir = data_artifact.download()

        model_artifact = run.use_artifact(args.model_name)
        run = model_artifact.logged_by()
        config_dict = run.json_config
        model_path = model_artifact.download(path_prefix=config_dict["model_name"])

        if "config_dir" in config_dict:
            config_artifact = run.use_artifact(
                f"{config_dict['config_dir']}:latest")
            config_dict["config_dir"] = config_artifact.download()

        df = pd.DataFrame(test(config_dict, audio_dir, wandb_logger, model_path), columns=["fname", "label"])

        artifact = wandb.Artifact(
            name=f"kaggle_results_{config_dict['model_name']}",
            type=ArtifactType.KAGGLE_RESULTS.value,
        )

        with artifact.new_file("kaggle_results.pth", mode="wb") as file:
            df.to_csv(file, sep=",", index=False)

        print(run.log_artifact(artifact))
