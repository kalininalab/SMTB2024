from argparse import Namespace

import pytest
import torch

from smtb.model import poolings

from .fixtures import sample_batch_x, sample_config


@pytest.mark.parametrize("pooling_layer", [x for x in poolings.values()])
def test_pooling(sample_batch_x, sample_config, pooling_layer):
    """Test the pooling layer."""
    # Create an instance of the pooling layer
    pooling = pooling_layer(sample_config)

    # Test the forward method
    output = pooling(sample_batch_x)
    assert output.shape == (sample_batch_x.shape[0], sample_batch_x.shape[-1])
