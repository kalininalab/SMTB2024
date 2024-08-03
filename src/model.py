from typing import Literal

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics as M
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Model(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        num_of_classes: int,
        task: Literal["regression", "binary_classification", "multiclass_classification"],
        dropout: float = 0.5,
        lr: float = 0.001,
        reduce_lr_patience: int = 50,
    ):  # you need to choose one of the tasks
        super().__init__()
        self.lr = lr
        self.reduce_lr_parience = reduce_lr_patience
        self.model = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.LazyLinear(num_of_classes),
        )
        if task == "binary_classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif task == "regression":
            self.loss_fn = nn.MSELoss()
        elif task == "multiclass_classification":
            self.loss_fn = nn.CrossEntropyLoss()

        self.metrics = self.get_metrics(self.task)

    def forward(self, x):
        return self.model(x).squeeze()

    def get_metrics(self):
        if self.task == "regression":
            m = M.MetricCollection(
                [
                    M.R2Score(num_outputs=1),
                    M.MeanSquaredError(),
                    M.PearsonCorrCoef(),
                    M.ConcordanceCorrCoef(),
                    M.ExplainedVariance(),
                ]
            )
        elif self.task == "binary_classification":
            m = M.MetricCollection([M.BinaryF1Score(), M.BinaryAUROC()])
        elif self.task == "multiclass_clasification":
            m = M.MetricCollection(
                [
                    M.classification.Precision(num_classes=self.num_of_classes, average="macro"),
                    M.classification.Recall(num_classes=self.num_of_classes, average="macro"),
                    M.classification.F1(num_classes=self.num_of_classes, average="macro"),
                ]
            )
        return {"train": m.clone(prefix="train/"), "val": m.clone(prefix="val/"), "test": m.clone(prefix="test/")}

    def shared_step(self, batch, name: str = "train"):
        x, y = batch
        y_pred = self.forward(x).float()
        y = y.float()
        loss = self.loss_fn(y_pred, y)
        self.metrics[name].update(y_pred, y)
        self.log(f"{name}/loss", loss)
        return loss

    def training_step(self, batch):
        return self.shared_step(batch, "train")

    def validation_step(self, batch):
        return self.shared_step(batch, "val")

    def test_step(self, batch):
        return self.shared_step(batch, "test")

    def shared_end(self, stage: "str"):
        metrics = self.metrics[stage].compute()
        self.log_dict(metrics)
        self.metrics[stage].reset()

    def on_train_epoch_end(self) -> None:
        self.shared_end("train")

    def on_validation_epoch_end(self) -> None:
        self.share_end("val")

    def on_test_epoch_end(self) -> None:
        self.shared_end("test")

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
