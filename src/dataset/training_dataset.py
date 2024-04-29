from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import lightning.pytorch as pl
import torch
import os
from typing import Callable

from settings import SplitType


class SpeechDataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        process_features: Callable[[Tensor], Tensor] | None = None
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.process_features = process_features

        self.train: TensorDataset | None = None
        self.val: TensorDataset | None = None
        self.test: TensorDataset | None = None

    def _load_dataset(self, path: str) -> TensorDataset:
        x, y = torch.load(path)
        y = y.squeeze(1)
        if self.process_features is not None:
            x = self.process_features(x)
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
        )