import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch.nn import functional as F
from torch import nn, Tensor
import torchmetrics
from torch.utils.data import Dataset
import lightning.pytorch as pl
import uuid


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        input_size: int,
        embedding_hidden: int,
        hidden_size: int,
        num_layers: int,
        n_tokens: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_type = "LSTM"
        self.embedding = Embedding(input_size, hidden_size, embedding_hidden)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(hidden_size, n_tokens)
        self.lr = lr

        self.init_weights()

        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_tokens, average="weighted"
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_tokens, average="weighted"
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_tokens, average="weighted"
        )

        self.best_model_name = ""
        self.lowest_valid_loss = float("inf")

        parent_dir = "run_checkpoints"
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        self.run_dir = os.path.join(parent_dir, f"runs_{uuid.uuid4().hex}")
        os.mkdir(self.run_dir)

    def _save_locally(self):
        path = os.path.join(self.run_dir, f"lstm_epoch_{self.current_epoch}.pth")
        torch.save(self.state_dict(), path)
        return path

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        src = self.embedding(src)
        output, (hidden, _) = self.lstm(src)
        output = output.mean(dim=1)
        output = self.linear(output)
        return output

    def loss(self, x, y):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return logits, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        preds, loss = self.loss(xs, ys)
        preds = torch.argmax(preds, 1)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.train_acc(preds, ys)
        self.log("train/accuracy", self.train_acc, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_acc(preds, ys)
        self.log(f"test/accuracy", self.test_acc, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.log(f"validation/loss", loss, on_epoch=True, on_step=False)
        self.valid_acc(preds, ys)
        self.log(f"validation/accuracy", self.valid_acc, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        path = self._save_locally()


class Embedding(nn.Module):
    def __init__(self, input_size: int, d_model: int, hidden_layers: int):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.linear1 = nn.Linear(input_size, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.input_size)
        x = self.relu(self.linear1(x))
        return self.linear2(x).view(batch_size, -1, self.d_model)
