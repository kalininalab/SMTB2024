from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def collate_fn(batch: list[tuple[torch.Tensor, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for downstream tasks.

    Args:
        batch (list(tuple(torch.Tensor, float))): tuples where the first element is a tensor representing the
            embeddings and the second element is a float representing the label.

    Returns:
        tuple(torch.Tensor, torch.Tensor): The first tensor is the padded embeddings and the second tensor is the
            labels.
    """
    tensors = [item[0].squeeze(0) for item in batch]
    floats = torch.tensor([item[1] for item in batch])
    padded_sequences = pad_sequence(tensors, batch_first=True, padding_value=0)
    return padded_sequences, floats


class DownstreamDataset(Dataset):
    """Dataset for downstream tasks. The `data_dir` is expected to have the following structure:
    data_dir
    ├── df.csv
    ├── prot_0.pt
    ├── prot_1.pt
    ├── ...
    └── prot_n.pt
    """

    def __init__(self, data_dir: str | Path, layer_num: int):
        """Initialize the DownstreamDataset."""
        self.data_dir = Path(data_dir)
        self.layer_num = layer_num
        assert self.data_dir.exists(), f"{self.data_dir} does not exist."
        assert self.data_dir.is_dir(), f"{self.data_dir} is not a directory."
        self.df = pd.read_csv(data_dir / "df.csv").dropna()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        """Return the embeddings and label for a given index."""
        embeddings = torch.load(self.data_dir / f"prot_{idx}.pt", weights_only=False)["representations"][
            self.layer_num
        ]
        label = self.df.iloc[idx]["value"]
        return embeddings, label

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.df.shape[0]


class DownstreamDataModule(L.LightningDataModule):
    """DataModule for downstream tasks. The `data_dir` is expected to have the following structure:
    data_dir
    ├── train
    │   ├── df.csv
    │   ├── prot_0.pt
    │   ├── prot_1.pt
    │   ├── ...
    │   └── prot_n.pt
    ├── valid
    │   ├── df.csv
    │   ├── prot_0.pt
    │   ├── prot_1.pt
    │   ├── ...
    │   └── prot_n.pt
    └── test
        ├── df.csv
        ├── prot_0.pt
        ├── prot_1.pt
        ├── ...
        └── prot_n.pt
    """

    def __init__(self, data_dir: str | Path, layer_num: int, batch_size: int, num_workers: int = 8):
        """Initialize the DownstreamDataModule."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        """Create train, val, test datasets."""
        self.train = DownstreamDataset(self.data_dir / "train", self.layer_num)
        self.valid = DownstreamDataset(self.data_dir / "valid", self.layer_num)
        self.test = DownstreamDataset(self.data_dir / "test", self.layer_num)

    def _get_dataloader(self, dataset: DownstreamDataset, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Create a DataLoader for a given dataset."""
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self._get_dataloader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return self._get_dataloader(self.valid)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return self._get_dataloader(self.test)
