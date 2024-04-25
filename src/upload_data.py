import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchvision.transforms as transforms
import os
import wandb
import json
from dataclasses import dataclass
import glob
import shutil

from settings import PROJECT, ENTITY, JobType, ArtifactType

DATA_DIR = "../data"


def load_annotation_file(path: str) -> set[str]:
    paths = set()
    with open(path, "r") as file:
        for line in file.readlines():
            paths.add(line)

    return paths


def split_background_noises(
        path: str, target_directory: str, target_length: float = 1.0
):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a valid directory")

    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    os.mkdir(target_directory)

    for noise_path in glob.glob(os.path.join(path, "*.wav")):
        metadata = torchaudio.info(noise_path)
        waveform, sample_rate = torchaudio.load(noise_path)
        num_samples = int(sample_rate*target_length)
        for i in range(0, waveform.shape[-1], num_samples):
            if i+num_samples > waveform.shape[-1]:
                break

            new_path = os.path.join(
                target_directory,
                f"{os.path.splitext(os.path.basename(noise_path))[0]}_{i // num_samples}.wav",
            )

            torchaudio.save(
                new_path,
                waveform[:, i:i+num_samples],
                sample_rate,
                encoding=metadata.encoding,
                bits_per_sample=metadata.bits_per_sample,
            )


class CustomAudioDataset(Dataset):
    def __init__(
            self, validation_labels_path: str, test_labels_path: str, audio_dir: str
    ):
        self.validation_files = load_annotation_file(validation_labels_path)
        self.test_files = load_annotation_file(test_labels_path)
        self.audio_dir = audio_dir

        self.audio_files = []
        for audio_file in glob.glob(os.path.join(audio_dir, "**", "*.wav")):
            pass



def load_data():
    with wandb.init(
            project=PROJECT, entity=ENTITY, job_type=JobType.UPLOAD_DATA.value
    ) as run:
        artifact = wandb.Artifact(
            "cinic-data",
            type=ArtifactType.DATASET.value,
            description="Raw CINIC dataset split into train/valid/test",
        )

        for split in ["train", "valid", "test"]:
            with artifact.new_file(split + ".pt", mode="wb") as file:
                dataset = ImageFolder(
                    os.path.join(DATA_DIR, split),
                    transform=transforms.ToTensor(),
                )
                dataloader = DataLoader(
                    dataset, batch_size=len(dataset), shuffle=False, num_workers=4
                )
                images, labels = next(iter(dataloader))

                # Save the tensors to a file
                torch.save((images, labels), file)

        # Upload to W&B
        run.log_artifact(artifact)


if __name__ == "__main__":
    load_data()
