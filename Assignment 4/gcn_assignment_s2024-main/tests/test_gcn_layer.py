
import unittest
import torch

from data.constants import TASK_CLASSIFY
from modeling.core.layers import GCNLayer
from src.data.graph import Graph
from src.data.utils import seed_all

seed_all(0)  # important! Otherwise tests won't be deterministic. They still may be stochastic though because of the randomness of the torch library.

# python -m unittest tests/test_gcn_model.py

class TestGCNLayer(unittest.TestCase):
    args = {
        "graph": "karate",
        "basepath": "data/",
        "task": TASK_CLASSIFY,
        "test_frac": 0.1,
        "val_frac": 0.1,
        "gpu": False
    }

    def test_forward(self):
        graph = Graph(**self.args)
        layer = GCNLayer(in_features=graph.d_features, out_features=32)
        output = layer(input_features=graph.features, adj=graph.adj)
        assert output.shape == (graph.n_nodes, 32)  # check output shape
        assert torch.isclose(output.sum(), torch.tensor(-46.9062), atol=1e-4)  # check output sum
        
