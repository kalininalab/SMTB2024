import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from src.data import DownstreamDataset
from src.model import Model

# Technical setting to make sure parallelization works if multiple models are trained in parallel
torch.multiprocessing.set_sharing_strategy("file_system")


def train(
    model_name: str,
    layer_num: int,
    hidden_dim: int = 512,
    batch_size: int = 1024,
    max_epoch: int = 200,
    dropout: float = 0.2,
    dataset: str = "/shared/stability",
    early_stopping_patience: int = 30,
    lr: float = 0.001,
    reduce_lr_patience: int = 30,
    seed: int = 42,
):
    """
    Trains the model with the specified parameters.

    Args:
        model_name (str): Name of the model.
        layer_num (int): Number of layers in the model.
        hidden_dim (int, optional): Hidden dimension of the model. Defaults to 512.
        batch_size (int, optional): Batch size. Defaults to 1024.
        max_epoch (int, optional): Maximum number of epochs. Defaults to 200.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        dataset (str, optional): Path to the dataset. Defaults to "/shared/stability".
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 30.
        lr (float, optional): Learning rate. Defaults to 0.001.
        reduce_lr_patience (int, optional): Patience for reducing the learning rate. Defaults to 30.
        seed (int, optional): Seed for reproducibility. Defaults to 42.
    """
    # For reproducibility
    seed_everything(seed)

    # Define the logger
    logger = WandbLogger(
        log_model=True,
        project=dataset.split("/")[-1],
        entity="smtb2024",
        name=f"{model_name.split('_')[1]}_{layer_num:02d}",
        config={
            "model_name": model_name,
            "layer_num": layer_num,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "early_stopping_patience": early_stopping_patience,
            "lr": lr,
            "reduce_lr_patience": reduce_lr_patience,
        },
    )

    # Define the callbacks with EarlyStopping and two more for nicer tracking
    callbacks = [
        EarlyStopping(monitor="val/loss", patience=early_stopping_patience, mode="min"),
        RichModelSummary(),
        RichProgressBar(),
        ModelCheckpoint(monitor="val/loss", mode="min"),
    ]

    # Define the Trainer and its most important arguments
    trainer = Trainer(
        devices=1,
        max_epochs=max_epoch,
        callbacks=callbacks,
        logger=logger,
    )

    # Initialize the model
    model = Model(hidden_dim=hidden_dim, dropout=dropout)

    # Look into the directory below
    datasets = []
    # TODO FIXME
    # for ds in ["train", "validation", "test"]:
    #     p = Path(dataset) / model_name / ds
    #     datasets.append(train_validation_test(p, layer_num))

    train_dataset = DownstreamDataset(datasets[0][0], datasets[0][1])
    validation_dataset = DownstreamDataset(datasets[1][0], datasets[1][1])
    test_dataset = DownstreamDataset(datasets[2][0], datasets[2][1])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # Fit and test the (best) model
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)


model_names = {
    48: "esm2_t48_15B_UR50D",
    36: "esm2_t36_3B_UR50D",
    33: "esm2_t33_650M_UR50D",
    30: "esm2_t30_150M_UR50D",
    12: "esm2_t12_35M_UR50D",
    6: "esm2_t6_8M_UR50D",
}


def run(
    num_layers: int,
    dataset: str,
    hidden_dim: int = 512,
    batch_size: int = 1024,
    max_epoch: int = 200,
    dropout: float = 0.2,
    early_stopping_patience: int = 30,
    lr: float = 0.001,
    reduce_lr_patience: int = 30,
    seed: int = 42,
):
    """
    Runs the training of the model with the given parameters.

    Args:
        num_layers (int): Number of layers in the model.
        dataset (str): Path to the dataset. Must contain folders with ESM embeddings.
        hidden_dim (int, optional): Hidden dimension of the model. Defaults to 512.
        batch_size (int, optional): Batch size. Defaults to 1024.
        max_epoch (int, optional): Maximum number of epochs. Defaults to 200.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 30.
        lr (float, optional): Learning rate. Defaults to 0.001.
        reduce_lr_patience (int, optional): Patience for reducing the learning rate. Defaults to 30.
        seed (int, optional): Seed for reproducibility. Defaults to 42.
    """
    model_name = model_names[num_layers]
    for layer in range(num_layers + 1):
        train(
            model_name,
            layer,
            hidden_dim,
            batch_size,
            max_epoch,
            dropout,
            dataset,
            early_stopping_patience,
            lr,
            reduce_lr_patience,
            seed,
        )
        wandb.finish()


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(run)
