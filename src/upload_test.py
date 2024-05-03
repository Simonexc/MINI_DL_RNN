import argparse
import os
from typing import Type
from torch.utils.data import DataLoader
import torch

import wandb

from settings import DATA_DIR, PROJECT, ENTITY, JobType, ArtifactType
import dataset.audio_dataset as audio_dataset


def upload_test_data(dataset: str):
    audio_dir = os.path.join(DATA_DIR, "test", "audio")

    dataset_class: Type[audio_dataset.BaseAudioDataset] = getattr(audio_dataset, dataset)
    dataset = dataset_class(audio_dir)
    dataloader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=4
    )
    audio, file_names = next(iter(dataloader))

    with wandb.init(
            project=PROJECT, entity=ENTITY, job_type=JobType.UPLOAD_DATA.value
    ) as run:
        artifact = wandb.Artifact(
            f"speech-{dataset_class.NAME}",
            type=ArtifactType.DATASET.value,
            description=dataset_class.DESCRIPTION,
        )
        with artifact.new_file("test.pt", mode="wb") as file:
            torch.save((audio, file_names), file)

        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process raw audio files from W&B and upload them back to W&B"
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Choose class for dataset processing from dataset.audio_dataset module. "
             "Write name of the class.",
    )

    # Parse the arguments
    args = parser.parse_args()

    upload_test_data(args.dataset)
