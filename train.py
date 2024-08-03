import torch

# import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split

from src.data import DownstreamDataset, load_dataset
from src.model import Model

# technical setting to make sure, parallelization works if multiple models are trained in parallel
torch.multiprocessing.set_sharing_strategy("file_system")


def train(
    dataset: DownstreamDataset,
    dataset_path: str,
    model_name: str,
    layer_num: int,
    hidden_dim: int = 512,
    batch_size: int = 1024,
    max_epoch: int = 200,
    dropout: float = 0.2,
    early_stopping_patience: int = 30,
    lr: float = 0.001,
    reduce_lr_patience: int = 30,
    seed: int = 42,
):
    # for reproducibility
    seed_everything(seed)

    # define the logger
    logger = CSVLogger(
        save_dir="/scratch/logs",
        name=f"{dataset_path.split('_')[1]}_{layer_num:02d}",
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
    # define the callbacks with EarlyStopping and two more for nicer tracking
    callbacks = [
        EarlyStopping(monitor="val/loss", patience=early_stopping_patience, mode="min"),
        RichProgressBar(),
        ModelCheckpoint(monitor="val/loss", mode="min"),
    ]

    # define the Trainer and it's most important arguments
    trainer = Trainer(
        devices=1,
        max_epochs=max_epoch,
        callbacks=callbacks,
        logger=logger,
    )

    # initialize the model
    model = Model(hidden_dim=hidden_dim, dropout=dropout)
    print(model)
    train, val, test = random_split(dataset, [0.7, 0.2, 0.1])

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(val, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # fit and test the (best) model
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
    dataset_path: str,
    hidden_dim: int = 512,
    batch_size: int = 1024,
    max_epoch: int = 200,
    dropout: float = 0.2,
    early_stopping_patience: int = 30,
    lr: float = 0.001,
    reduce_lr_patience: int = 30,
    seed: int = 42,
):
    """Runs the training of the model with the given parameters.

    Args:
        num_layers (int): Models with which number of layers to use.
        dataset (str): Path to the dataset. Must contain folders with esm embeddings.
        hidden_dim (int, optional): Hidden dimension of the model. Defaults to 512.
        batch_size (int, optional): Batch size. Defaults to 1024.
        max_epoch (int, optional): Maximum number of epochs. Defaults to 10000.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 100.
        lr (float, optional): Learning rate. Defaults to 0.001.
        reduce_lr_patience (int, optional): Patience for reducing the learning rate. Defaults to 50.
        seed (int, optional): Seed for reproducibility. Defaults to 42.
    """
    model_name = model_names[num_layers]
    for i, dataset in enumerate(load_dataset(dataset_path, num_layers)):
        print("Train layer", i)
        train(
            dataset,
            dataset_path,
            model_name,
            i,
            hidden_dim,
            batch_size,
            max_epoch,
            dropout,
            early_stopping_patience,
            lr,
            reduce_lr_patience,
            seed,
        )


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(run)
