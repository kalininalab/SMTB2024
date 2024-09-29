import argparse
import random
from pathlib import Path

import pandas as pd
import pytest
import torch


@pytest.fixture
def mock_data_dir(tmp_path: Path):
    # Create a temporary directory with mock data
    data_dir = tmp_path / "data"
    for ds in ["train", "valid", "test"]:
        ds_data_dir = data_dir / ds
        ds_data_dir.mkdir(parents=True)

        # Create a mock CSV file
        df = pd.DataFrame({"value": [0.1, 0.2, 0.3, 0.4]})
        df.to_csv(ds_data_dir / "df.csv", index=False)

        # Create mock .pt files
        for i in range(len(df)):
            # seq_len between 50 and 60, embedding_dim=320
            data = {"representations": {0: torch.rand((random.randint(50, 60), 320))}}
            torch.save(data, ds_data_dir / f"prot_{i}.pt")
    return data_dir


@pytest.fixture
def sample_batch_x():
    """Create a sample input tensor of shape (batch_size, seq_len, embedding_dim)"""
    return torch.randn(4, 10, 32)  # Example: batch_size=4, seq_len=10, embedding_dim=8


@pytest.fixture
def sample_config():
    """Create a sample config object for training."""
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
