import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.layers import GCNLayer


class GCN(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ):
        super(GCN, self).__init__()
        # TODO: add 2 layers of GCN
        self.gcn_1 = GCNLayer(in_features=input_dim, out_features=hidden_dim)
        self.gcn_2 = GCNLayer(in_features=hidden_dim, out_features=output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.sparse_coo) -> torch.Tensor:
        # given the input node features, and the adjacency matrix, run GCN
        # The order of operations should roughly be:
        # 1. Apply the first GCN layer
        # 2. Apply Relu
        # 3. Apply Dropout
        # 4. Apply the second GCN layer

        # TODO: your code here
        hidden = self.gcn_1(x, adj)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.gcn_2(hidden, adj)
        return output
