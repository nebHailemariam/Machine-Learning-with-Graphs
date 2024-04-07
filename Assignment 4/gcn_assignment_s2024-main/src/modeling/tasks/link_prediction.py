import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.gcn import GCN


class LinkPrediction(nn.Module):
    def __init__(self, hidden_dim: int):
        """Link prediction module.
        We want to predict the edge label (0 or 1) for each edge.
        We assume that the model gets the node features as input (i.e., GCN is already applied to the node features).
        Args:
            hidden_dim (int): [The hidden dimension of the GCN layer (i.e., feature dimension of the nodes)]
        """

        super(LinkPrediction, self).__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.edge_classifier = nn.Linear(
            hidden_dim, 2
        )  # TODO: Define the edge classifier

    def forward(
        self,
        node_features_after_gcn: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:

        # node_features_after_gcn: [num_nodes, hidden_dim]
        # edges: [2, num_edges]
        # the function should return classifier logits for each edge
        # Note that the output should not be probabilities, rather one logit for each class (so the output should be batch_size x 2).
        # TODO: Implement the forward pass of the link prediction module
        head = node_features_after_gcn[edges[0]]
        tail = node_features_after_gcn[edges[1]]
        head_tail_concat = torch.cat([head, tail], dim=1)
        out = self.linear(head_tail_concat)
        out = self.relu(out)
        out = self.edge_classifier(out)
        return out
