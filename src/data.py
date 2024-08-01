from typing import Any, Literal


import esm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle as pkl


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
    def __init__(self, num_layers: Literal[6, 12, 30, 33]):
        self.num_layers = num_layers
        self.models = {
            6: "esm2_t6_8M_UR50D",
            12: "esm2_t12_35M_UR50D",
            30: "esm2_t30_150M_UR50D",
            33: "esm2_t33_650M_UR50D",
        }

        if self.num_layers not in self.models:
            raise ValueError(
                f"Unsupported number of layers: {self.num_layers}. Supported layers are: {list(self.models.keys())}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = getattr(esm.pretrained, self.models[self.num_layers])()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(self.device)

    def run(
        self,
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
        results = []
        for prot in tqdm(data):
            batch_labels, batch_strs, batch_tokens = self.batch_converter([('alper', prot)])
            batch_tokens = batch_tokens.to(self.device)  # Ensure tokens are on the GPU
            with torch.no_grad():
                i = self.model.forward(batch_tokens, repr_layers=layers, return_contacts=contacts)
                detached_i = {}
                for k, v in i.items():
                    if isinstance(v, dict):  # Check if value is a dictionary (like "representations")
                        detached_i[k] = {k1: v1.detach().cpu() for k1, v1 in v.items()}
                    else:
                        detached_i[k] = v.detach().cpu()
                results.append(detached_i)
        return results

class DataRead:
    def get_protlist(df):
        data = pd.read_csv(df)
        prot_list = []
        d_dict = data.to_dict(orient='index')
        for key in d_dict.keys():
                n = d_dict[key]['primary']
                prot_list.append(n)
        return prot_list

    def embeddings_to_dataset(dataframe, embeddings, layer):
        labels = list(dataframe[dataframe.columns[1]])
        embedd_list = []
        for i in range(len(embeddings)):
            embedd_list.append(embeddings[i]['representations'][layer])
        return DownstreamDataset(embedd_list, labels)