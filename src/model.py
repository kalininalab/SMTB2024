import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as M
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Model(pl.LightningModule):
    """
    A PyTorch Lightning Module for a simple neural network model.

    Args:
        hidden_dim (int): The number of hidden dimensions for the model.
        dropout (float): Dropout rate for regularization. Default is 0.5.
        lr (float): Learning rate for the optimizer. Default is 0.001.
        reduce_lr_patience (int): Patience parameter for the ReduceLROnPlateau scheduler. Default is 50.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.5, lr: float = 0.001, reduce_lr_patience: int = 50):
        super().__init__()
        self.lr = lr
        self.reduce_lr_patience = reduce_lr_patience
        self.model = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.LazyLinear(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x).squeeze(1)

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], name: str = "train") -> torch.Tensor:
        """
        A shared step for training, validation, and testing.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of data containing inputs and targets.
            name (str): The name of the step (train, val, test). Default is "train".

        Returns:
            torch.Tensor: The computed loss.
        """
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
        """
        A single training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of data containing inputs and targets.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        A single validation step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of data containing inputs and targets.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        A single test step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of data containing inputs and targets.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler.

        Returns:
            tuple: A tuple containing the optimizers and the learning rate schedulers.
        """
        optimizers = [optim.Adam(self.parameters(), lr=self.lr)]
        schedulers = [
            {
                "scheduler": ReduceLROnPlateau(
                    optimizers[0],
                    factor=0.1,
                    patience=self.reduce_lr_patience,
                    min_lr=1e-7,
                ),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizers, schedulers
