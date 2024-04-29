from torch.utils.data import TensorDataset, DataLoader, default_collate
from torch import Tensor
import lightning.pytorch as pl
import torch
import os
from typing import Callable

from settings import SplitType
from .feature_processors import BaseProcessor


class SpeechDataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        feature_processor: BaseProcessor | None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.feature_processor = feature_processor

        self.train: TensorDataset | None = None
        self.val: TensorDataset | None = None
        self.test: TensorDataset | None = None

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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=self._process_features(),
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