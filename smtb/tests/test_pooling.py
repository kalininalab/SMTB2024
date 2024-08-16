import pytest
import torch

from smtb.model import poolings


@pytest.fixture
def sample_input():
    # Create a sample input tensor of shape (batch_size, seq_len, embedding_dim)
    return torch.randn(4, 10, 8)  # Example: batch_size=4, seq_len=10, embedding_dim=8


@pytest.mark.parametrize("pooling_layer", [x for x in poolings.values()])
def test_pooling(sample_input, pooling_layer):
    # Create an instance of the pooling layer
    pooling = pooling_layer(input_dim=sample_input.shape[-1])

    # Test the forward method
    output = pooling(sample_input)
    assert output.shape == (sample_input.shape[0], sample_input.shape[-1])


if __name__ == "__main__":
    pytest.main()
