from pathlib import Path

import pytest
import torch

from smtb.data import DownstreamDataModule, DownstreamDataset

from .fixtures import mock_data_dir


def test_downstream_dataset(mock_data_dir: Path):
    dataset = DownstreamDataset(mock_data_dir / "train", layer_num=0)

    # Check the length of the dataset
    assert len(dataset) == 4

    # Check the first item
    embeddings, label = dataset[0]
    assert isinstance(embeddings, torch.Tensor)
    assert isinstance(label, float)
    assert embeddings.size(1) == 320


def test_downstream_data_module(mock_data_dir: Path):
    data_module = DownstreamDataModule(mock_data_dir, layer_num=0, batch_size=2, num_workers=0)
    data_module.setup()

    # Check the train, valid, and test datasets
    assert len(data_module.train) == 4
    assert len(data_module.valid) == 4
    assert len(data_module.test) == 4

    # Check the train dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    embeddings, labels = batch
    assert embeddings.size(0) == 2
    assert labels.size(0) == 2
    assert embeddings.size(2) == 320
    assert 50 <= embeddings.size(1) <= 60
