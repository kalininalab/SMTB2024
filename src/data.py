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
        self.df = pd.read_csv(data_dir / "df.csv")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        embeddings = torch.load(self.data_dir / f"prot_{idx}.pt")["representations"][self.layer_num]
        label = self.df.iloc[idx]["value"]
        return embeddings, label


def get_protlist(df: str):
    """
    Get the protein sequences from the given DataFrame.

    :param df: Path to the DataFrame
    :return: A list of protein sequences
    """
    data = pd.read_csv(df)
    prot_list = []
    d_dict = data.to_dict(orient="index")
    for key in d_dict.keys():
        n = d_dict[key][list(d_dict[key].keys())[0]]
        prot_list.append(n)
    return prot_list


model_names = {
    48: "esm2_t48_15B_UR50D",
    36: "esm2_t36_3B_UR50D",
    33: "esm2_t33_650M_UR50D",
    30: "esm2_t30_150M_UR50D",
    12: "esm2_t12_35M_UR50D",
    6: "esm2_t6_8M_UR50D",
}


# def load_dataset(nlayers: int, dataset_name: str):
#     root = Path("/") / "scratch" / "data_roman" / dataset_name  # TODO FIXME
#     model_name = model_names[nlayers]
#     for split in ["train", "val", "test"]:
#         output_path = root / "processed" / model_name / split
#         # output_path.mkdir(exist_ok=True, parents=True)
#         print(root / "raw" / f"{split}.csv")
#         protlist = get_protlist(root / "raw" / f"{split}.csv")
#         fasta_path = root / f"{split}.fasta"
#         with open(fasta_path, "w") as f:
#             for i, seq in enumerate(protlist):
#                 print(f">Prot{i:06d}", seq, sep="\n", file=f)
#         print(f"COMMAND: python -m src.extract {model_name} {output_path} {str(output_path)} --include per_tok")
#         os.system(
#             f"python -m src.extract {model_name} {str(fasta_path)} {str(output_path)} --include per_tok"
#         )  # TODO FIXME
