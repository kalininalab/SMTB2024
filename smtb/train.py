import argparse
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

import wandb
from smtb.data import DownstreamDataModule
from smtb.model import RegressionModel


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
    wandb.finish()
