from typing import Literal

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as M
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.pooling import GlobalAttentionPooling, MeanPooling

poolings = {"mean": MeanPooling, "attention": GlobalAttentionPooling}


class Model(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 512,
        pooling: Literal["mean", "attention"] = "attention",
        dropout: float = 0.5,
        lr: float = 0.001,
        reduce_lr_patience: int = 10,
    ):
        super().__init__()
        self.lr = lr
        self.reduce_lr_parience = reduce_lr_patience
        self.pooling = pooling
        self.model = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            poolings[pooling](hidden_dim),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.LazyLinear(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], name: str = "train") -> torch.Tensor:
        x, y = batch

        # compute the prediction
        y_pred = self.forward(x).float()
        y = y.float()

        # compute the loss
        loss = F.mse_loss(y_pred, y)

        # compute and log the metrics
        self.log(f"{name}/loss", loss)
        self.log(f"{name}/r2", M.functional.r2_score(y_pred, y))
        self.log(f"{name}/pearson", M.functional.pearson_corrcoef(y_pred, y))
        self.log(f"{name}/expvar", M.functional.explained_variance(y_pred, y))
        self.log(f"{name}/concord", M.functional.concordance_corrcoef(y_pred, y))
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimisers = [optim.Adam(self.parameters(), lr=self.lr)]
        schedulers = [
            {
                "scheduler": ReduceLROnPlateau(
                    optimisers[0],
                    factor=0.1,
                    patience=10,
                    min_lr=1e-7,
                ),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimisers, schedulers
