from abc import ABC, abstractmethod
import glob
import os

import torch
import torchaudio
from torch.utils.data import Dataset


from settings import SplitType
from .utils import get_class_id_from_audio_path


class BaseAudioDataset(Dataset, ABC):
    NAME: str
    DESCRIPTION: str

    def __init__(
            self, audio_dir: str
    ):
        self.audio_dir = audio_dir

        self.audio_files = self.get_audio()

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.audio_files[idx][0]
        waveform, sample_rate = torchaudio.load(path)
        waveform = self.process_audio(waveform, sample_rate)
        class_id = self.audio_files[idx][1]

        return waveform, torch.tensor([class_id], dtype=torch.long)

    def get_audio(self) -> list[tuple[str, int]]:
        audio_files = []
        for audio_path in glob.glob(os.path.join(self.audio_dir, "*", "*.wav")):
            audio_files.append((audio_path, get_class_id_from_audio_path(audio_path)))

        return audio_files

    @abstractmethod
    def process_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Convert audio waveform (loaded with torchaudio.load) into correct format.
        """


class WaveformAudioDataset(BaseAudioDataset):
    """
    Dataset for loading audio files as waveform tensors with set length of 1 second
    (truncated or padded with zeros if needed).
    """
    NAME = "waveform"
    DESCRIPTION = "Waveform audio dataset split into train/valid/test"

    def process_audio(self, waveform: torch.Tensor, _sample_rate: int) -> torch.Tensor:
        waveform = waveform.squeeze()
        if waveform.shape[-1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[-1]))
        elif waveform.shape[-1] > 16000:
            waveform = waveform[:16000]
        return waveform.squeeze()
