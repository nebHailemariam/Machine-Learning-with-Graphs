import logging
import time
from itertools import chain
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.modeling.tasks.node_classification import NodeClassifier
from src.data.utils import seed_all
from src.data.graph import Graph
from src.data.utils import accuracy, batchify_edges
from src.modeling.tasks.link_prediction import LinkPrediction
from src.training.args import get_training_args
from src.data.constants import *

logging.basicConfig(level=logging.INFO)



def main():
    args = get_training_args()
    seed_all(args.seed)
    trainer = Trainer(args)
    trainer.train()


class Trainer(object):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        self.graph = Graph(**vars(self.args))
        self.init_model()
        self.optimizer = optim.Adam(
            chain(self.link_prediction_model.parameters(), self.model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def init_model(self):
        # Ιnitializes a node classification model and a link prediction model
        # The node classification model is jointly trained with the link prediction model.
        self.model = NodeClassifier(
            input_dim=self.graph.features.shape[1],
            hidden_dim=self.args.hidden,
            n_classes=self.graph.labels.max().item() + 1,
            dropout=self.args.dropout,
        ).to(self.device)
        print(self.model)
        if self.args.task ==  TASK_LINK_PRED:
            self.link_prediction_model = LinkPrediction(hidden_dim=self.args.hidden,).to(
                self.device
            )
            self.loss_func = nn.CrossEntropyLoss()
            print(self.link_prediction_model)
        else:
            raise NotImplemented(f"The current implementation does not support {self.args.task}")

    def train(self):
        # Train model
        t_total = time.time()
        for epoch_idx in range(self.args.epochs):
            self.epoch(epoch_idx)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        self.test()

    def epoch(self, epoch):
        # runs a single epoch
        t = time.time()
        

        # run one epoch of GCN to refine node_embeddings
        for batch_idx, (batch_nodes, batch_edges, batch_labels) in enumerate(
            batchify_edges(
                self.graph.train_edges, self.graph.train_edge_labels, batch_size=512, shuffle=True
            )
        ):
            self.link_prediction_model.train()
            self.model.train()
            self.optimizer.zero_grad()

            total_loss = 0.0

            if batch_idx % self.args.node_classification_interval == 0:
                node_classification_output = self.model(self.graph.features, self.graph.adj)
                node_classifcation_loss = F.nll_loss(
                    node_classification_output[self.graph.idx_train],
                    self.graph.labels[self.graph.idx_train],
                )
                node_classification_accuracy = accuracy(
                    node_classification_output[self.graph.idx_train],
                    self.graph.labels[self.graph.idx_train],
                )
                total_loss += node_classifcation_loss

            x = self.model(self.graph.features, self.graph.adj, classify=False)

            logits = self.link_prediction_model(x, batch_edges)
            edge_classification_loss = self.loss_func(logits, batch_labels)
            edge_classification_accuracy = accuracy(logits, batch_labels)
            total_loss += edge_classification_loss

            total_loss.backward()
            self.optimizer.step()

        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(total_loss.item()),
            "node_classifcation_loss: {:.4f}".format(node_classifcation_loss.item()),
            "edge_classification_loss: {:.4f}".format(edge_classification_loss.item()),
            "edge_acc: {:.4f}".format(edge_classification_accuracy.item()),
            "node_acc: {:.4f}".format(node_classification_accuracy.item()),
            "time: {:.4f}s".format(time.time() - t),
        )
        self.eval("dev", self.graph.val_edges, self.graph.val_edge_labels)

    def test(self):
        self.eval("test", self.graph.test_edges, self.graph.test_edge_labels)

    def eval(self, split, edges, labels):
        # Evaluates the model on the given split
        avg_accuracy, avg_loss = 0.0, 0.0
        n_items = 0.0
        self.link_prediction_model.eval()
        self.model.eval()
        for batch_nodes, batch_edges, batch_labels in batchify_edges(
            edges, labels, batch_size=512, shuffle=False
        ):
            with torch.no_grad():
                x = self.model(self.graph.features, self.graph.adj, classify=False)
                logits = self.link_prediction_model(x, batch_edges)

                avg_loss += F.cross_entropy(logits, batch_labels).item() * batch_edges.shape[1]
                avg_accuracy += accuracy(logits, batch_labels).item() * batch_edges.shape[1]
                n_items += batch_edges.shape[1]

        avg_accuracy /= n_items
        avg_loss /= n_items
        print(
            f"{split}_n_items: {n_items}",
            f"{split}_acc: {avg_accuracy:.4f}",
            f"{split}_loss: {avg_loss:.4f}",
        )


if __name__ == "__main__":
    main()
