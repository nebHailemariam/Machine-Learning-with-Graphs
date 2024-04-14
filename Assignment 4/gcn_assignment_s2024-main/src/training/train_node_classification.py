import logging
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.data.utils import seed_all, accuracy
from src.modeling.tasks.node_classification import NodeClassifier
from src.training.args import get_training_args
from src.data.graph import Graph
from src.data.constants import *
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Training settings


def main():
    args = get_training_args()
    seed_all(args.seed)
    trainer = Trainer(args)
    trainer.train()
    trainer.plot()


class Trainer(object):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
        )

        self.graph = Graph(**vars(self.args))

        self.init_model()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def init_model(self):

        self.model = NodeClassifier(
            input_dim=self.graph.features.shape[1],
            hidden_dim=self.args.hidden,
            n_classes=self.graph.labels.max().item() + 1,
            dropout=self.args.dropout,
        ).to(self.device)
        logging.info("Number of classes: {}".format(self.graph.labels.max().item() + 1))

    def train(self):
        # Train model
        t_total = time.time()
        losses = []
        L = 4
        for epoch_idx in range(self.args.epochs):
            res = self.epoch(epoch_idx)
            losses.append((res[0].item(), res[1].item()))

        # self.plot_results(losses)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # Writing losses to CSV file
        with open("losses.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(["Epoch", "L", "Loss", "Acc"])
            for (
                idx,
                loss,
            ) in enumerate(zip(losses)):
                writer.writerow([idx + 1, L, loss[0][0], loss[0][1]])
        # Testing
        self.test()
        # Group the data by 'L' value

    def plot(self):
        loss_df = pd.read_csv("losses.csv")
        grouped = loss_df.groupby("L")

        # Plotting
        plt.figure(figsize=(10, 5))
        for L, group in grouped:
            plt.plot(group["Epoch"], group["Loss"], label=f"L = {L}")

        plt.title("Loss vs. Epoch with Different L")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plotting
        plt.figure(figsize=(10, 5))
        for L, group in grouped:
            plt.plot(group["Epoch"], group["Acc"], label=f"L = {L}")

        plt.title("Accuracy vs. Epoch with Different L")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def epoch(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        # note: the node representations themselves are not being updated; only the W matrix is.
        # what happens if you also update the node features?
        output = self.model(self.graph.features, self.graph.adj)
        loss_train = F.cross_entropy(
            output[self.graph.idx_train], self.graph.labels[self.graph.idx_train]
        )
        acc_train = accuracy(
            output[self.graph.idx_train], self.graph.labels[self.graph.idx_train]
        )
        loss_train.backward()
        self.optimizer.step()

        if self.args.run_validation:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.graph.features, self.graph.adj)

        loss_val = F.cross_entropy(
            output[self.graph.idx_val], self.graph.labels[self.graph.idx_val]
        )
        acc_val = accuracy(
            output[self.graph.idx_val], self.graph.labels[self.graph.idx_val]
        )
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss_train.item()),
            "acc_train: {:.4f}".format(acc_train.item()),
            "loss_val: {:.4f}".format(loss_val.item()),
            "acc_val: {:.4f}".format(acc_val.item()),
            "time: {:.4f}s".format(time.time() - t),
        )
        return loss_train, acc_train

    def plot_results(self, loss):

        plt.figure(figsize=(10, 5))

        # Plotting loss
        plt.plot(loss)
        plt.xlabel("Number of Layers (L)")
        plt.ylabel("Loss")
        plt.title("Loss vs. Number of Layers")

        plt.show()

    def test(self):
        self.model.eval()
        output = self.model(self.graph.features, self.graph.adj)
        loss_test = F.cross_entropy(
            output[self.graph.idx_test], self.graph.labels[self.graph.idx_test]
        )
        acc_test = accuracy(
            output[self.graph.idx_test], self.graph.labels[self.graph.idx_test]
        )
        print(
            "Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
        )


if __name__ == "__main__":
    main()
