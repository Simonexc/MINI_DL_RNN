import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import wandb
import json
from dataclasses import dataclass
import numpy as np
import glob
import shutil
from pathlib import Path
import uuid

from settings import PROJECT, ENTITY, JobType, ArtifactType, AUDIO_FILE_METADATA, SplitType, DATA_DIR
from dataset.utils import load_annotation_file, get_class_name_from_audio_path


def split_background_noises(
        path: str, target_directory: str, target_length: float = 1.0  # in seconds
) -> list[str]:
    """
    Split background noises into smaller chunks of target_length.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a valid directory")

    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    os.mkdir(target_directory)

    silence_paths = []

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

            silence_paths.append(new_path)

    return silence_paths


def assert_audio_metadata(audio_path: str):
    assert os.path.isfile(audio_path), f"{audio_path} does not exist"
    audio_metadata = torchaudio.info(audio_path)
    for key, value in AUDIO_FILE_METADATA.items():
        assert getattr(audio_metadata, key, None) == value, f"Metadata mismatch for {audio_path}, key: {key}, expected value: {value}, actual value: {getattr(audio_metadata, key, None)}"


def upload_raw_data():
    audio_dir = os.path.join(DATA_DIR, "train", "audio")
    validation_files = load_annotation_file(os.path.join(DATA_DIR, "train", "validation_list.txt"), audio_dir)
    test_files = load_annotation_file(os.path.join(DATA_DIR, "train", "testing_list.txt"), audio_dir)

    # split silence in to train/validation/test with 0.8/0.1/0.1 ratio
    silence_paths = np.array(split_background_noises(
        os.path.join(audio_dir, "_background_noise_"),
        os.path.join(audio_dir, "silence"),
    ))
    # split silence paths into validation and test
    np.random.shuffle(silence_paths)
    data_split_size = int(len(silence_paths) * 0.1)
    validation_files.update(set(silence_paths[:data_split_size]))
    test_files.update(set(silence_paths[:data_split_size]))

    with wandb.init(
            project=PROJECT, entity=ENTITY, job_type=JobType.UPLOAD_DATA.value
    ) as run:
        artifact = wandb.Artifact(
            "speech-raw",
            type=ArtifactType.DATASET.value,
            description="Raw audio file dataset split into train/valid/test",
        )

        for split in [SplitType.TRAIN, SplitType.VALIDATION, SplitType.TEST]:
            if split == SplitType.TRAIN:
                files = set(glob.glob(os.path.join(audio_dir, "*", "*.wav")))
                files -= validation_files
                files -= test_files
            elif split == SplitType.VALIDATION:
                files = validation_files
            else:
                files = test_files

            for file in files:
                if "_background_noise_" in file:
                    continue
                try:
                    assert_audio_metadata(file)
                except (AssertionError, RuntimeError):
                    continue

                class_name = get_class_name_from_audio_path(file)
                artifact.add_file(file, name=f"{split.value}/{class_name}/{uuid.uuid4().hex}.wav")

        # Upload to W&B
        run.log_artifact(artifact)


if __name__ == "__main__":
    upload_raw_data()
