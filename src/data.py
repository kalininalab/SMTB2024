import os
from pathlib import Path
from typing import Any, Literal

import esm
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DownstreamDataset(Dataset):
    def __init__(self, embeddings: list[torch.Tensor], labels: list):
        """
        Initialize the DownstreamDataset as a collection of the embeddings and labels.
        :param embeddings: A list of embeddings.
        :param labels: A list of labels.
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]


class ESMEmbedder:
    def __init__(self, num_layers: Literal[6, 12, 30, 33, 36, 48]):
        self.num_layers = num_layers
        self.models = {
            48: "esm2_t48_15B_UR50D",
            36: "esm2_t36_3B_UR50D",
            33: "esm2_t33_650M_UR50D",
            30: "esm2_t30_150M_UR50D",
            12: "esm2_t12_35M_UR50D",
            6: "esm2_t6_8M_UR50D",
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = getattr(esm.pretrained, self.models[self.num_layers])()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(self.device)

    def run(
        self,
        dataset: str,
        data: list[str],
        layers: list[int] | None = None,
        contacts: bool = False,
    ) -> list[dict[Any, dict[Any, Any] | Any]]:
        """
        Compute the embeddings from one ESM Model for the given protein sequences.
        :param data: A list of protein sequences, give as strings.
        :param layers: A list of layers to look at. If none, all layers are used.
        :param contacts: Boolean flag to extract contacts (mostly not used)
        :return: A list of dictionaries with the embeddings for each protein sequence.
        """
        if layers is None:
            layers = range(self.num_layers + 1)
        for i, prot in tqdm(enumerate(data)):
            batch_labels, batch_strs, batch_tokens = self.batch_converter([("alper", prot)])
            batch_tokens = batch_tokens.to(self.device)  # Ensure tokens are on the GPU
            with torch.no_grad():
                i = self.model.forward(batch_tokens, repr_layers=layers, return_contacts=contacts)
            for k, v in i.items():
                if isinstance(v, dict):  # Check if value is a dictionary (like "representations")
                    embedding = {k1: v1.detach().cpu() for k1, v1 in v.items()}
                else:
                    embedding = v.detach().cpu()


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


def embeddings_to_dataset(dataframe: pd.DataFrame, embeddings: list[dict[Any, dict[Any, Any] | Any]], layer: int):
    """
    Convert the embeddings to a DownstreamDataset object.

    :param dataframe: DataFrame of the dataset
    :param embeddings: List of embeddings for each protein sequence
    :param layer: Layer of the model to use for the embeddings
    :return: DownstreamDataset object
    """
    labels = list(dataframe[dataframe.columns[1]])
    embedd_list = []
    for i in range(len(embeddings)):
        embedd_list.append(embeddings[i]["representations"][layer].mean([0, 1]))
    return DownstreamDataset(embedd_list, labels)


model_names = {
    48: "esm2_t48_15B_UR50D",
    36: "esm2_t36_3B_UR50D",
    33: "esm2_t33_650M_UR50D",
    30: "esm2_t30_150M_UR50D",
    12: "esm2_t12_35M_UR50D",
    6: "esm2_t6_8M_UR50D",
}


def load_dataset(nlayers: int, dataset_name: str):
    """
    Preprocess the dataset from the given path and compute the embeddings for the model with nlayers-many layers.

    :param path: Path to the dataset
    :param nlayers: Maximum number of layers of the model to use for embedding
    :return: A list of DownstreamDataset objects, one for each layer of the model
    """
    root = Path("/") / "scratch" / "data_roman" / dataset_name
    model_name = model_names[nlayers]
    for split in ["train", "val", "test"]:
        print("embedding", split)
        output_path = root / "processed" / model_name / split
        # output_path.mkdir(exist_ok=True, parents=True)
        print(root / "raw" / f"{split}.csv")
        protlist = get_protlist(root / "raw" / f"{split}.csv")
        fasta_path = root / f"{split}.fasta"
        with open(fasta_path, "w") as f:
            for i, seq in enumerate(protlist):
                print(f">Prot{i:06d}", seq, sep="\n", file=f)
        print(f"COMMAND: python -m src.extract {model_name} {output_path} {str(output_path)} --include per_tok")
        os.system(f"python -m src.extract {model_name} {str(fasta_path)} {str(output_path)} --include per_tok")
