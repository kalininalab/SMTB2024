import csv
import os
from pathlib import Path

import torch
import torchmetrics as M

# import wandb
from src.data import DownstreamDataModule
from src.model import Model

torch.multiprocessing.set_sharing_strategy("file_system")


def log_metrics_to_csv(metrics, file_path):
    fieldnames = metrics.keys()

    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writerow(metrics)


def train(
    dataset_path: str,
    model_name: str,
    pooling: str,
    layer_num: int,
    dataset_name: str,
    batch_size: int = 1024,
    num_workers: int = 12,
):
    # exm: /scratch/data/fluorescence/processed/logs/esm2_t6_8M_UR50D_L0_mean_roman/version_0/checkpoints/
    checkpoint_path = (
        Path("/scratch/data") / dataset_name / "processed/logs" / (model_name + f"_L{layer_num}_{pooling}_roman")
    )
    if len(os.listdir(str(checkpoint_path))) == 0:
        return
    checkpoint_path = checkpoint_path / os.listdir(str(checkpoint_path))[0] / "checkpoints"
    if len(os.listdir(str(checkpoint_path))) == 0:
        return
    checkpoint_path = checkpoint_path / os.listdir(str(checkpoint_path))[0]

    metrics_path = Path("/scratch/data") / dataset_name / "processed/logs" / "test_metrics.csv"

    model = Model(pooling=pooling)
    checkpoint = torch.load(str(checkpoint_path))
    model.load_state_dict(checkpoint["state_dict"])

    datamodule = DownstreamDataModule(dataset_path, layer_num, batch_size, num_workers)
    datamodule.setup()
    model.eval()

    m = M.MetricCollection(
        [
            M.R2Score(num_outputs=1),
            M.MeanSquaredError(),
            M.PearsonCorrCoef(),
            M.ConcordanceCorrCoef(),
            M.ExplainedVariance(),
        ]
    )

    for x, y in datamodule.test_dataloader():
        y_pred = model(x)
        m.update(y_pred, y)

    metrics = m.compute()
    # print(metrics)
    log_metrics_to_csv(metrics, file_path=metrics_path)
    m.reset()


dataset_names = ["fluorescence", "stability"]

model_names = {
    6: "esm2_t6_8M_UR50D",
    12: "esm2_t12_35M_UR50D",
    30: "esm2_t30_150M_UR50D",
}

pooling_options = ["mean", "attention"]

for dataset in dataset_names:
    for model in model_names.keys():
        dataset_path = Path("/scratch/data")
        dataset_path = dataset_path / dataset / "processed" / model_names[model]

        for layer_num in range(model):
            for pooling in pooling_options:
                train(str(dataset_path), model_names[model], pooling, layer_num, dataset)
                # exit(0)
