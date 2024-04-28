from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import torch
import os


class SpeechDataset(pl.LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def _load_dataset(self, path: str) -> TensorDataset:
        x, y = torch.load(path)
        y = y.squeeze(1)
        return TensorDataset(x, y)

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.train = self._load_dataset(os.path.join(self.dataset_dir, "train.pt"))
            self.val = self._load_dataset(os.path.join(self.dataset_dir, "validation.pt"))
        if stage == 'test' or stage is None:
            self.test = self._load_dataset(os.path.join(self.dataset_dir, "test.pt"))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
        )