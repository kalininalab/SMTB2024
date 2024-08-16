from argparse import Namespace

import torch
import torch.nn as nn


class BasePooling(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pools the input tensor.
        All pooling layers should implement this method.

        Args:
            x (torch.Tensor): Tensor of size (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: Tensor of size (batch_size, embedding_dim)
        """
        raise NotImplementedError


class GlobalAttentionPooling(BasePooling):
    def __init__(self, config: Namespace):
        super().__init__(config)

        self.linear_key = nn.Linear(self.config.hidden_dim, 1)  # Project to a single dimension
        self.linear_query = nn.Linear(self.config.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.linear_key(x)
        queries = self.linear_query(x)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = self.softmax(attention_scores)
        pooled_output = torch.matmul(attention_weights, x).sum(dim=1)
        return pooled_output


class MeanPooling(BasePooling):
    def __init__(self, config: Namespace):
        super().__init__(config)

    def forward(self, x):
        return x.mean(dim=1)
