import torchaudio.transforms as T
from torch.utils.data import TensorDataset, DataLoader, default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from torch import Tensor
import torch
import os

from settings import SplitType
from .feature_processors import BaseProcessor

import lightning.pytorch as pl


class SpeechDataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        feature_processor: BaseProcessor | None,
        train_num_samples: int | None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.feature_processor = feature_processor
        self.train_num_samples = train_num_samples

        self.train: TensorDataset | None = None
        self.val: TensorDataset | None = None
        self.test: TensorDataset | None = None

        self.train_weights: Tensor | None = None

    @staticmethod
    def _load_dataset(path: str) -> TensorDataset:
        x, y = torch.load(path)
        y = y.squeeze(1)
        return TensorDataset(x, y)

    def setup(self, stage: str | None = None):
        if stage == 'fit' or stage is None:
            self.train = self._load_dataset(
                os.path.join(self.dataset_dir, f"{SplitType.TRAIN.value}.pt")
            )
            self.train_weights = (1 / self.train.tensors[1].unique(return_counts=True)[1])[
                self.train.tensors[1]
            ]
            self.val = self._load_dataset(
                os.path.join(self.dataset_dir, f"{SplitType.VALIDATION.value}.pt")
            )
        if stage == 'test' or stage is None:
            self.test = self._load_dataset(
                os.path.join(self.dataset_dir, f"{SplitType.TEST.value}.pt")
            )

    def _process_features(self):
        def collate_fn(batch):
            x, y = default_collate(batch)
            if self.feature_processor is not None:
                x = self.feature_processor(x)
            return x, y
        return collate_fn

    def train_collate_fn(self, batch):
        # Define transforms
        spectrogram = T.Spectrogram(n_fft=400, win_length=400, hop_length=160, center=False, power=None)  # Returns complex spectrogram
        amplitude_to_db = T.AmplitudeToDB()  # Converts magnitude to dB scale
        time_stretch = T.TimeStretch(n_freq=201, fixed_rate=1.2) ##Te mozna pozmieniac
        freq_mask = T.FrequencyMasking(freq_mask_param=30)  ##Te mozna pozmieniac
        time_mask = T.TimeMasking(time_mask_param=100)  ##Te mozna pozmieniac
        
        processed_batch = []
        for (waveform, label) in batch:
            # Check if waveform needs to be unsqueezed
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            
            # Convert waveform to complex spectrogram
            spec = spectrogram(waveform)
            
            # Apply Time Stretch
            stretched_spec = time_stretch(spec)
            
            # Convert complex spectrogram to magnitude for masking
            magnitude_spec = torch.abs(stretched_spec)
            
            # Convert magnitude to dB scale (if needed by your model or other transformations)
            db_spec = amplitude_to_db(magnitude_spec)
            
            # Apply frequency and time masks
            masked_spec = freq_mask(db_spec)
            masked_spec = time_mask(masked_spec)
            
            # If necessary, convert back to waveform or other required format
            # waveform = your_inverse_transform_here(masked_spec)
            
            if self.feature_processor:
                waveform = self.feature_processor(waveform)
            
            processed_batch.append((waveform, label))

        return default_collate(processed_batch)



    def train_dataloader(self) -> DataLoader:
        sampler = WeightedRandomSampler(
            self.train_weights,
            self.train_num_samples or self.train.tensors[1].shape[0],
            replacement=True,
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=4,
            sampler=sampler,
            collate_fn=self.train_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            collate_fn=self._process_features(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            collate_fn=self._process_features(),
        )
