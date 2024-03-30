import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """Implements one layer of a GCN. A GCN model would typically have ~2/3 such stacked layers.
    """
    def __init__(self, in_features: int, out_features: int):
        """Initializes a GCN layer.
        Args:
            in_features (int): [Input feature dimensions.]
            out_features (int): [Output feature dimensions.]
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, input_features: torch.Tensor, adj: torch.sparse_coo):
        """Given a matrix of input nodes (input), and a sparse adjacency matrix, 
        implements the GCN.
        Args:
            input_features (torch.Tensor): [The input nodes matrix]
            adj (torch.sparse_coo): [The sparse adjacency matrix]

        Returns:
            [type]: [description]
        """
        # TODO: Implement GCN layer.
        # Two steps are required:
        # 1. Apply the linear transformation to the input features.
        # 2. Multiply the output of the linear transformation by the adjacency matrix.
        # Hint: You can use torch.spmm() to do the matrix multiplication. The implementation 
        # should be about 2-3 lines of code.
        # Hint: You can use the unit test (tests/test_gcn_layer.py) to check your implementation.
        
        # YOUR CODE HERE
        return output