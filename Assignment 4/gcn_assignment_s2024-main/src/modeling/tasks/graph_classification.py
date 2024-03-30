import torch
import torch.nn as nn


class GraphClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, pooling_op: str):
        super(GraphClassifier, self).__init__()

        self.graph_classifier = # TODO: Define the graph classifier
        # graph classifier can be an MLP
        self.pooling_op = pooling_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given node features x, applies two operations:
        1. Pools the node representations using the pooling operation specified
        2. Applies a classifier to the pooled representation
        """
        # TODO: Implement the forward pass for the graph classifier
        return classifier_logits
    
    def pool(self, x: torch.Tensor, operation: str = "last") -> torch.Tensor:
        """Given node features x, applies a pooling operation to return a 
        single aggregated feature vector.

        Args:
            x (torch.Tensor): [The node features]
            operation (str, optional): [description]. Defaults to "last".

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [A single feature vector for the graph]
        """
        if operation == "mean":
            return x.mean(dim=0)
        elif operation == "max":
            return x.max(dim=0)[0]
        elif operation == "last":
            return x[-1]
        else:
            raise NotImplementedError()