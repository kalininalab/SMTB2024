from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DownstreamDataset(Dataset):
    def __init__(self, data_dir: str | Path, layer_num: int):
        self.data_dir = Path(data_dir)
        self.layer_num = layer_num
        assert self.data_dir.exists(), f"{self.data_dir} does not exist."
        assert self.data_dir.is_dir(), f"{self.data_dir} is not a directory."
        self.df = pd.read_csv(data_dir / "df.csv")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        embeddings = torch.load(self.data_dir / f"prot_{idx}.pt")["representations"][self.layer_num]
        label = self.df.iloc[idx]["value"]
        return embeddings, label


class DownstreamDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str | Path, layer_num: int, batch_size: int, num_workers: int = 8):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.num = num_workers

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train = DownstreamDataset(self.data_dir / "train", self.layer_num)
            self.valid = DownstreamDataset(self.data_dir / "valid", self.layer_num)
        if stage == "test" or stage is None:
            self.test = DownstreamDataset(self.data_dir / "valid", self.layer_num)

    def _get_dataloader(self, dataset: DownstreamDataset, shuffle: bool = False) -> torch.utils.data.DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.valid)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test)
