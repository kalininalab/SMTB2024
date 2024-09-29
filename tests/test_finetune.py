import os

from smtb.train import train
from .fixtures import mock_data_dir, sample_config


def test_train(sample_config, mock_data_dir):
    os.environ["WANDB_MODE"] = "dryrun"
    data_dir = mock_data_dir
    sample_config.dataset_path = data_dir

    train(sample_config)
