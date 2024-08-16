from argparse import Namespace

import pytest
import torch

from smtb.model import RegressionModel


@pytest.fixture
def sample_input():
    # Create a sample input tensor of shape (batch_size, seq_len, embedding_dim)
    return torch.randn(4, 10, 8)  # Example: batch_size=4, seq_len=10, embedding_dim=8


@pytest.fixture
def sample_config():
    return Namespace(hidden_dim=256, pooling="mean", dropout=0.3, lr=0.01, reduce_lr_patience=5)


def test_regression_model_forward(sample_config, sample_input):
    model = RegressionModel(sample_config)
    output = model.forward(sample_input)
    assert output.size(0) == 4


if __name__ == "__main__":
    pytest.main()
