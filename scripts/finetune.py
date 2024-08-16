import argparse
from pathlib import Path

import torch

# import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from smtb.data import DownstreamDataModule
from smtb.model import RegressionModel

# technical setting to make sure, parallelization works if multiple models are trained in parallel
torch.multiprocessing.set_sharing_strategy("file_system")


def train(config: argparse.Namespace):
    dataset_path = Path(config.dataset_path)
    seed_everything(config.seed)
    logger = WandbLogger()
    callbacks = [
        EarlyStopping(monitor="val/loss", patience=config.early_stopping_patience, mode="min"),
        RichProgressBar(),
        ModelCheckpoint(monitor="val/loss", mode="min"),
    ]

    # define the Trainer and it's most important arguments
    trainer = Trainer(
        devices=-1,
        max_epochs=config.max_epoch,
        callbacks=callbacks,
        logger=logger,
    )
    model = RegressionModel(config)
    datamodule = DownstreamDataModule(dataset_path, config.layer_num, config.batch_size, config.num_workers)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")  # good solution !!!!
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--layer_num", type=int, required=True)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reduce_lr_patience", type=int, default=10)
    parser.add_argument("--reduce_lr_factor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    config = parser.parse_args()
    train(config)
