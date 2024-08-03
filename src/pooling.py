import torch
import torch.nn as nn


class GlobalAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):  # No need for hidden_dim for global pooling
        super(GlobalAttentionPooling, self).__init__()

        self.linear_key = nn.Linear(input_dim, 1)  # Project to a single dimension
        self.linear_query = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Calculate keys and queries (note the single dimension)
        keys = self.linear_key(x)  # (batch_size, seq_len, 1)
        queries = self.linear_query(x)

        # Attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # No scaling needed
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_len, seq_len)

        # Global weighted sum (sum over sequence length)
        pooled_output = torch.matmul(attention_weights, x).sum(dim=1)  # (batch_size, embedding_dim)

        return pooled_output


class MeanPooling(nn.Module):
    def forward(x):
        return x.mean(1)
