# placeholder class for the graph data structure
import logging
import torch
from src.data.utils import load_data
from src.data.constants import *

logging.basicConfig(level=logging.INFO)


class Graph:
    # Wraps a graph
    def __init__(
        self,
        graph: str,
        basepath: str,
        task: str,
        test_frac: float,
        val_frac: float,
        gpu: bool,
        **kwargs,
    ):
        """Wraps a graph

        Args:
            graph (str): Name of the graph (e.g., cora)
            basepath (str): Path to the data folder. The data folder should have a graph folder with cites and content files. See f"data/cora/" for example.
            task (str): The task to be performed. Should be one of [TASK_LINK_PRED, TASK_CLASSIFY]
            test_frac (float): Fraction of samples for the test split.
            val_frac (float): Fraction of samples for the validation split.
            gpu (bool): Whether to use gpu or not.
        """
        self.graph = graph
        self.basepath = basepath
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.task = task
        self.device = torch.device("cuda" if gpu else "cpu")
        self.load_graph()


    def load_graph(self):
        (
            self.adj,  # sparse adjacency matrix (torch.sparse_coo)
            self.features,  # (num_nodes, feature_dim)
            self.labels,  # (num_nodes, num_classes)
            self.idx_train,
            self.idx_val,
            self.idx_test,
        ) = load_data(
            graph=self.graph,
            path=self.basepath,
            test_frac=self.test_frac,
            val_frac=self.val_frac,
        )
        self.features.to(self.device)
        self.adj.to(self.device)
        self.labels.to(self.device)
        self.idx_train.to(self.device)
        self.idx_val.to(self.device)
        self.idx_test.to(self.device)
        self.d_features = self.features.shape[1]
        self.n_nodes = self.features.shape[0]
        if self.task == TASK_LINK_PRED:  # need to add edges for the link prediction task
            logging.info("Loading edges")
            self.add_edges()

    def add_edges(self):
        def _edge_negative_sample(edges):
            return torch.stack(
                [edges[0, torch.randperm(edges.shape[1])], edges[1, torch.randperm(edges.shape[1])]]
            )

        # step 1: split edges into train, val, and test
        edges = self.adj.coalesce().indices()  # (2, num_edges)
        num_edges = edges.shape[1]
        num_test_edges, num_val_edges = int(self.test_frac * num_edges), int(
            self.val_frac * num_edges
        )
        num_train_edges = num_edges - num_test_edges - num_val_edges
        self.train_edges_positive, self.test_edges_positive, self.val_edges_positive = torch.split(
            edges, [num_train_edges, num_test_edges, num_val_edges], dim=-1
        )

        # step 2: add negative examples by randomly permuting the positive edges
        self.train_edges_negative = _edge_negative_sample(self.train_edges_positive)
        self.test_edges_negative = _edge_negative_sample(self.test_edges_positive)
        self.val_edges_negative = _edge_negative_sample(self.val_edges_positive)

        # step 3: combine the positive and the negative edges
        self.train_edge_labels = torch.cat(
            [
                torch.ones(self.train_edges_positive.shape[1]),
                torch.zeros(self.train_edges_negative.shape[1]),
            ],
            dim=0,
        ).long()
        self.test_edge_labels = torch.cat(
            [
                torch.ones(self.test_edges_positive.shape[1]),
                torch.zeros(self.test_edges_positive.shape[1]),
            ],
            dim=0,
        ).long()
        self.val_edge_labels = torch.cat(
            [
                torch.ones(self.val_edges_positive.shape[1]),
                torch.zeros(self.val_edges_positive.shape[1]),
            ],
            dim=0,
        ).long()

        self.train_edges = torch.cat([self.train_edges_positive, self.train_edges_negative], dim=-1)
        self.test_edges = torch.cat([self.test_edges_positive, self.test_edges_negative], dim=-1)
        self.val_edges = torch.cat([self.val_edges_positive, self.val_edges_negative], dim=-1)