import unittest

import torch
from data.constants import TASK_CLASSIFY
from modeling.core.gcn import GCN
from modeling.core.layers import GCNLayer
from src.data.graph import Graph
from src.data.utils import seed_all
# important! Otherwise tests won't be deterministic. They still may be stochastic though because of the randomness of the torch library.
seed_all(0)

args = {
    "graph": "karate",
    "basepath": "data/",
    "task": TASK_CLASSIFY,
    "test_frac": 0.1,
    "val_frac": 0.1,
    "gpu": False
}


class TestGCN(unittest.TestCase):

    def test_forward(self):
        graph = Graph(**args)
        model1 = GCN(graph.d_features, hidden_dim=32, output_dim=8, dropout=0.5)
        output1 = model1(x=graph.features, adj=graph.adj)
        assert output1.shape == (graph.n_nodes, 8)
        assert torch.isclose(output1.sum(), torch.tensor(3.4986),  atol=1e-4)

        