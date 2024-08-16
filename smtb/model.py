from argparse import Namespace

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as M
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .pooling import GlobalAttentionPooling, MeanPooling

poolings = {"mean": MeanPooling, "attention": GlobalAttentionPooling}


class BaseModel(pl.LightningModule):
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimisers = [optim.Adam(self.parameters(), lr=self.config.lr)]
        schedulers = [
            {
                "scheduler": ReduceLROnPlateau(
                    optimisers[0],
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                ),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimisers, schedulers


class RegressionModel(BaseModel):
    def __init__(
        self,
        config: Namespace,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.LazyLinear(config.hidden_dim),
            poolings[config.pooling](config.hidden_dim),
            nn.LazyLinear(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.LazyLinear(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], name: str = "train") -> torch.Tensor:
        x, y = batch
        y_pred = self.forward(x).float()
        y = y.float()
        loss = F.mse_loss(y_pred, y)
        self.log(f"{name}/loss", loss)
        self.log(f"{name}/r2", M.functional.r2_score(y_pred, y))
        self.log(f"{name}/pearson", M.functional.pearson_corrcoef(y_pred, y))
        self.log(f"{name}/expvar", M.functional.explained_variance(y_pred, y))
        self.log(f"{name}/concord", M.functional.concordance_corrcoef(y_pred, y))
        return loss
