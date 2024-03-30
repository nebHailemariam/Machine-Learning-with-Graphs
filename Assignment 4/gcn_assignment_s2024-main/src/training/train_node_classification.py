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


# Training settings


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
        for epoch_idx in range(self.args.epochs):
            self.epoch(epoch_idx)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        self.test()

    def epoch(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        # note: the node representations themselves are not being updated; only the W matrix is.
        # what happens if you also update the node features?
        output = self.model(self.graph.features, self.graph.adj)
        loss_train = F.nll_loss(output[self.graph.idx_train], self.graph.labels[self.graph.idx_train])
        acc_train = accuracy(output[self.graph.idx_train], self.graph.labels[self.graph.idx_train])
        loss_train.backward()
        self.optimizer.step()

        if self.args.run_validation:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.graph.features, self.graph.adj)

        loss_val = F.nll_loss(output[self.graph.idx_val], self.graph.labels[self.graph.idx_val])
        acc_val = accuracy(output[self.graph.idx_val], self.graph.labels[self.graph.idx_val])
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss_train.item()),
            "acc_train: {:.4f}".format(acc_train.item()),
            "loss_val: {:.4f}".format(loss_val.item()),
            "acc_val: {:.4f}".format(acc_val.item()),
            "time: {:.4f}s".format(time.time() - t),
        )

    def test(self):
        self.model.eval()
        output = self.model(self.graph.features, self.graph.adj)
        loss_test = F.nll_loss(output[self.graph.idx_test], self.graph.labels[self.graph.idx_test])
        acc_test = accuracy(output[self.graph.idx_test], self.graph.labels[self.graph.idx_test])
        print(
            "Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
        )


if __name__ == "__main__":
    main()