# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset_path", type=str, required=True)
# parser.add_argument("--layer_num", type=int, required=True)
# parser.add_argument("--pooling", type=str, default="mean")
# parser.add_argument("--hidden_dim", type=int, default=512)
# parser.add_argument("--batch_size", type=int, default=1024)
# parser.add_argument("--num_workers", type=int, default=12)
# parser.add_argument("--max_epoch", type=int, default=1000)
# parser.add_argument("--dropout", type=float, default=0.2)
# parser.add_argument("--early_stopping_patience", type=int, default=20)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--reduce_lr_patience", type=int, default=10)
# parser.add_argument("--reduce_lr_factor", type=float, default=0.1)
# parser.add_argument("--seed", type=int, default=42)

import argparse
import os

import pytest

from ..train import train


@pytest.fixture
def sample_config():
    return argparse.Namespace(
        layer_num=0,
        pooling="mean",
        hidden_dim=32,
        batch_size=2,
        num_workers=0,
        max_epoch=2,
        dropout=0.2,
        early_stopping_patience=20,
        lr=0.001,
        reduce_lr_patience=10,
        reduce_lr_factor=0.1,
        seed=42,
    )


def test_train(sample_config, mock_data_dir):
    os.environ["WANDB_MODE"] = "dryrun"
    data_dir = mock_data_dir
    sample_config.dataset_path = data_dir

    train(sample_config)
