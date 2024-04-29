import os
import uuid

import torch
from torch.nn import functional as F
from torch import nn, Tensor
import torchmetrics
import lightning.pytorch as pl
import numpy as np
import wandb

from settings import NUM_CLASSES, ArtifactType
from dataset.feature_processors import BaseProcessor


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        lr: float,
        upload_best_model: bool = True,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.lr = lr
        self.upload_best_model = upload_best_model

        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=NUM_CLASSES,
            average="weighted",
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=NUM_CLASSES,
            average="weighted",
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=NUM_CLASSES,
            average="weighted",
        )

        self.valid_losses = []
        self.test_losses = []

        # Model
        self.best_model_name = ""
        self.lowest_valid_loss = float("inf")
        self.lowest_valid_epoch: int | None = None

        parent_dir = "run_checkpoints"
        if not os.path.exists("run_checkpoints"):
            os.mkdir(parent_dir)
        self.run_dir = os.path.join(parent_dir, f"runs_{uuid.uuid4().hex}")
        os.mkdir(self.run_dir)

    def _save_local(self):
        path = os.path.join(self.run_dir, f"epoch_{self.current_epoch}.pth")
        torch.save(self.state_dict(), path)

        return path

    def _save_remote(self, filename: str, **metadata):
        artifact = wandb.Artifact(
            name=filename,
            type=ArtifactType.MODEL.value,
            metadata=metadata
        )

        with artifact.new_file(filename + ".pth", mode="wb") as file:
            torch.save(self.state_dict(), file)

        return self.logger.experiment.log_artifact(artifact)

    def load_local(self, model_path: str):
        self.load_state_dict(torch.load(model_path))

    def load_remote(self, model_name: str):
        artifact = self.logger.use_artifact(model_name)
        model_file_name = model_name[:model_name.rfind(":")] + ".pth"
        model_path = artifact.download(path_prefix=model_file_name)

        self.load_local(os.path.join(model_path, model_file_name))

    def load_best_model(self):
        self.load_local(self.best_model_name)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        return logits, loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

    def on_validation_epoch_start(self):
        self.valid_losses = []

    def on_test_epoch_start(self):
        self.test_losses = []

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        xs, ys = batch
        preds, loss = self.loss(xs, ys)
        preds = torch.argmax(preds, 1)

        self.log('train/loss', loss, on_epoch=True, on_step=True)
        self.train_acc(preds, ys)
        self.log('train/accuracy', self.train_acc, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_losses.append(loss.cpu())

        self.log(f"validation/loss", loss, on_epoch=True, on_step=False)
        self.valid_acc(preds, ys)
        self.log(f'validation/accuracy', self.valid_acc, on_epoch=True, on_step=False)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.test_losses.append(loss.cpu())

        self.log(f"test/loss", loss, on_epoch=True, on_step=False)
        self.test_acc(preds, ys)
        self.log(f'test/accuracy', self.test_acc, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        path = self._save_local()

        avg_loss = np.mean(self.valid_losses)
        if avg_loss < self.lowest_valid_loss:
            self.lowest_valid_epoch = self.current_epoch
            self.lowest_valid_loss = avg_loss
            self.best_model_name = path

    def on_test_end(self):
        avg_loss = np.mean(self.test_losses)

        if self.upload_best_model:
            self._save_remote(
                self.model_name, epoch=self.lowest_valid_epoch, loss=avg_loss
            )
