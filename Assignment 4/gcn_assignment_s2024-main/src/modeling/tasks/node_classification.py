import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.gcn import GCN


class NodeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        """The node classifier module.
            we need two components for the node classification task:
            1. a GCN model, that can learn informative features for each node
            2. a linear classifier, that takes the features for each node and uses them to predict the node label
        Args:
            input_dim (int): [Input feature dimensions (i.e. the number of features for each node)]
            hidden_dim (int): [Hidden dimensions for the GCN model]
            n_classes (int): [Number of classes for the node classification task]
            dropout (float): [Dropout rate for the GCN model]
        """
        super(NodeClassifier, self).__init__()

        self.gcn = GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )  # TODO: initialize the GCN model
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, n_classes),
        )  # TODO: initialize the linear classifier

    def forward(
        self, x: torch.Tensor, adj: torch.sparse_coo, classify: bool = True
    ) -> torch.Tensor:
        # TODO: implement the forward pass of the node classification task
        if classify:
            output = self.gcn(x, adj)
            output = self.node_classifier(output)
            softmax = nn.Softmax(dim=1)

            return softmax(output)
        else:
            output = self.gcn(x, adj)
            return output
