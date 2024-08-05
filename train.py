import random
import string
from pathlib import Path

import torch

# import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger

from src.data import DownstreamDataModule
from src.model import Model

# technical setting to make sure, parallelization works if multiple models are trained in parallel
torch.multiprocessing.set_sharing_strategy("file_system")


def random_string(k: int = 5):
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))


def train(
    dataset_path: str,
    model_name: str,
    pooling: str,
    layer_num: int,
    random_name: str,
    hidden_dim: int = 512,
    batch_size: int = 1024,
    num_workers: int = 12,
    max_epoch: int = 1000,
    dropout: float = 0.2,
    early_stopping_patience: int = 20,
    lr: float = 0.001,
    reduce_lr_patience: int = 10,
    seed: int = 42,
    gpu: int = 0,
):
    dataset_path = Path(dataset_path)
    seed_everything(seed)
    print(dataset_path.parents[0] / "logs")
    logger = CSVLogger(
        save_dir=dataset_path.parents[0] / "logs",
        name=f"{model_name}_L{layer_num}_{pooling}_{random_name}",
    )

    logger.log_hyperparams(
        {
            "model_name": model_name,
            "layer_num": layer_num,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "early_stopping_patience": early_stopping_patience,
            "lr": lr,
            "reduce_lr_patience": reduce_lr_patience,
        }
    )
    callbacks = [
        EarlyStopping(monitor="val/loss", patience=early_stopping_patience, mode="min"),
        RichProgressBar(),
        ModelCheckpoint(monitor="val/loss", mode="min"),
    ]

    # define the Trainer and it's most important arguments
    trainer = Trainer(
        devices=[gpu],
        max_epochs=max_epoch,
        callbacks=callbacks,
        logger=logger,
    )
    model = Model(hidden_dim=hidden_dim, pooling=pooling, dropout=dropout)
    datamodule = DownstreamDataModule(dataset_path, layer_num, batch_size, num_workers)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(train)
