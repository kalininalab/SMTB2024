from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class DownstreamDataset(Dataset):
    def __init__(self, data_dir: str | Path, layer_num: int):
        self.data_dir = Path(data_dir)
        self.layer_num = layer_num
        assert self.data_dir.exists(), f"{self.data_dir} does not exist."
        assert self.data_dir.is_dir(), f"{self.data_dir} is not a directory."
        assert (self.data_dir / "df.csv").exists(), f"{self.data_dir/'df.csv'} does not exist."
        self.df = pd.read_csv(self.data_dir / "df.csv")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        embeddings = torch.load(self.data_dir / f"prot_{idx}.pt")["representations"][self.layer_num]
        label = self.df.iloc[idx]["log_fluorescence"]
        return embeddings, label
