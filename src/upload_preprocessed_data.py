import argparse
import torch
from torch.utils.data import DataLoader
import os
import wandb
from typing import Type

from settings import PROJECT, ENTITY, JobType, ArtifactType, SplitType
import dataset.audio_dataset as audio_dataset


def upload_data(dataset: str):
    with wandb.init(
            project=PROJECT, entity=ENTITY, job_type=JobType.UPLOAD_DATA.value
    ) as run:
        data_artifact = run.use_artifact("speech-raw:latest")
        audio_dir = data_artifact.download()

        dataset_class: Type[audio_dataset.BaseAudioDataset] = getattr(audio_dataset, dataset)
        artifact = wandb.Artifact(
            f"speech-{dataset_class.NAME}",
            type=ArtifactType.DATASET.value,
            description=dataset_class.DESCRIPTION,
        )

        for split in [SplitType.TRAIN, SplitType.VALIDATION, SplitType.TEST]:
            dataset = dataset_class(os.path.join(audio_dir, split.value))
            dataloader = DataLoader(
                dataset, batch_size=len(dataset), shuffle=False, num_workers=4
            )
            audio, labels = next(iter(dataloader))

            with artifact.new_file(split.value + ".pt", mode="wb") as file:
                torch.save((audio, labels), file)

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

    upload_data(args.dataset)
