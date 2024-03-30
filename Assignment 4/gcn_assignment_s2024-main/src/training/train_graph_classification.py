import logging
import time
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim

from src.modeling.core.gcn import GCN
from src.modeling.tasks.graph_classification import GraphClassifier
from src.data.utils import seed_all
from src.data.utils import accuracy
from src.training.args import get_training_args
from src.data.constants import *

logging.basicConfig(level=logging.INFO)


def main():
    args = get_training_args()
    seed_all(args.seed)
    trainer = GraphClassificationTrainer(args)
    trainer.train()


class GraphClassificationTrainer(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        self.read_graphs()
        if "ipynb" in args:
            return
        self.init_model()
        self.optimizer = optim.Adam(
            chain(self.graph_classification_model.parameters(), self.model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def read_graphs(self):
        graphs = torch.load(self.args.graphs_path)
        self.train_graphs = graphs["train"]
        self.test_graphs = graphs["test"]
        self.val_graphs = graphs["val"]
        self.graphs = {"train": self.train_graphs, "test": self.test_graphs, "val": self.val_graphs}
        self.input_dim = self.train_graphs[0]["node_features"].shape[1]
        logging.info(f"Input dim {self.input_dim}")

    def init_model(self):
        # Initialize a GCN and a graph classification model
        self.model = GCN(
            self.input_dim,
            hidden_dim=self.args.hidden,
            output_dim=self.args.hidden // 2,
            dropout=self.args.dropout,
        ).to(self.device)
        print(self.model)
        if self.args.task == TASK_GRAPH_CLASSIFICATION:
            self.graph_classification_model = GraphClassifier(
                hidden_dim=self.args.hidden // 2,
                num_classes=self.args.num_graph_classes,
                pooling_op=self.args.pooling_op,
            ).to(self.device)
            self.loss_func = nn.CrossEntropyLoss()
            print(self.graph_classification_model)
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
        self.eval("test")

    def epoch(self, epoch):
        # Runs one epoch of training and evaluation
        t = time.time()

        epoch_acc = 0.0
        epoch_loss = 0.0
        self.optimizer.zero_grad()

        # run one epoch of GCN to refine node_embeddings
        for graph_idx, graph in enumerate(self.train_graphs):
            self.graph_classification_model.train()
            self.model.train()

            graph_class_loss, graph_class_acc, graph_pred_label = self.run_graph_classification(
                graph
            )

            epoch_loss += graph_class_loss.item()

            epoch_acc += graph_class_acc

            graph_class_loss.backward()

            if (graph_idx + 1) % 5 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        epoch_loss /= len(self.train_graphs)
        epoch_acc /= len(self.train_graphs)

        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(epoch_loss),
            "graph_classification_acc: {:.4f}".format(epoch_acc),
            "time: {:.4f}s".format(time.time() - t),
        )
        self.eval("val")
        self.eval("test")

    def run_graph_classification(self, graph):
        # Runs one graph through the graph classification model
        x = self.model(graph["node_features"], graph["adj"])  # (num_nodes, hidden_dim)
        logits = self.graph_classification_model(x).unsqueeze(0)
        loss = self.loss_func(logits, graph["graph_label"])
        acc = accuracy(logits, graph["graph_label"])
        pred_label = logits.max(1)[1].type_as(graph["graph_label"]).item()
        return loss, acc, pred_label

    def eval(self, split: str):
        # Evaluates the model on the specified split
        avg_accuracy, avg_loss = 0.0, 0.0

        self.graph_classification_model.eval()
        self.model.eval()
        pred_labels = []
        true_labels = []
        for graph in self.graphs[split]:
            with torch.no_grad():
                loss, acc, pred_label = self.run_graph_classification(graph)

                avg_loss += loss
                avg_accuracy += acc

                pred_labels.append(pred_label)
                true_labels.append(graph["graph_label"])

        # calculate confusion matrix
        cm = torch.zeros(self.args.num_graph_classes, self.args.num_graph_classes)

        for t, p in zip(true_labels, pred_labels):
            cm[t, p] += 1

        # precision, recall, f1
        precision = cm.diag() / (cm.sum(1) + 1e-8)
        recall = cm.diag() / (cm.sum(0) + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"{split}_precision: {precision.mean().item() * 100:.2f}")
        print(f"{split}_recall: {recall.mean().item() * 100:.2f}")
        print(f"{split}_f1: {f1.mean().item() * 100:.2f}")

        n_items = len(self.graphs[split])
        avg_accuracy /= n_items
        avg_loss /= n_items
        print(
            f"{split}_n_items: {n_items}",
            f"{split}_acc: {avg_accuracy:.4f}",
            f"{split}_loss: {avg_loss:.4f}",
        )


if __name__ == "__main__":
    main()
