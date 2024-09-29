from argparse import Namespace

import pytest
import torch

from smtb.model import RegressionModel

from .fixtures import sample_batch_x, sample_config


def test_regression_model_forward(sample_config, sample_batch_x):
    model = RegressionModel(sample_config)
    output = model.forward(sample_batch_x)
    assert output.size(0) == sample_batch_x.size(0)
