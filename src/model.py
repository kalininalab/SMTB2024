import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as M
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Model(pl.LightningModule):
    def __init__(self, hidden_dim, num_of_classes: int, dropout: float = 0.5, lr: float = 0.001, reduce_lr_patience: int = 50, task = "regression"):
        super().__init__()
        self.lr = lr
        self.reduce_lr_parience = reduce_lr_patience
        self.model = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.LazyLinear(num_of_classes),
        )

    def forward(self, x):
        if self.task == "multiclass":
            return self.model(x)
        elif self.task == "binar":
            pass
        elif self.task == "regression":
            return self.model(x).squeeze(1)

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], name: str = "train") -> torch.Tensor:
        x, y = batch

        # compute the prediction
        
        y_pred = self.forward(x)
        y = y.float()

        # compute the loss
        if self.task == "multiclass":
            loss =  F.cross_entropy(y_pred, y)
        elif self.task == "binar":
            pass
        elif self.task == "regression":
            y_pred = self.forward(x).float()

        # compute and log the metrics
        self.log(f"{name}/loss", loss)

        if self.task == "multiclass":
            precision = M.classification.Precision(num_classes=self.num_of_classes, average='macro')
            recall = M.classification.Recall(num_classes=self.num_of_classes, average='macro')
            f1_score = M.classification.F1(num_classes=self.num_of_classes, average='macro')

            self.log(f"{name}/prec", precision)
            self.log(f"{name}/rec", recall)
            self.log(f"{name}/F1", f1_score)
        elif self.task == "binar":
            pass
        elif self.task == "regression":
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
                    patience=self.reduce_lr_parience,
                    min_lr=1e-7,
                ),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimisers, schedulers
