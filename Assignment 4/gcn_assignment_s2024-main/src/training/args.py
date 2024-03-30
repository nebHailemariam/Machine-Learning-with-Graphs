# the training arguments

import argparse


def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Whether to use GPU for training. Note that you should be able to run all the tasks for this assignment without a GPU.",
    )
    parser.add_argument("--graph", type=str, help="Graph tag.")
    parser.add_argument(
        "--basepath", type=str, default="data/", help="Directory that contains all the graphs."
    )
    parser.add_argument(
        "--graphs_path", type=str, help="Path to a pt file that contains the graphs that can be used for graph classification."
    )
    parser.add_argument("--task", type=str, help="The task to be performed.")
    parser.add_argument(
        "--run_validation", action="store_true", default=True, help="Validate during training pass."
    )
    parser.add_argument("--seed", type=int, default=10708, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train.")

    parser.add_argument("--node_classification_interval", type=int, default=5, help="Number of epochs to train.")


    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay (L2 loss on parameters)."
    )
    parser.add_argument("--hidden", type=int, default=128, help="Number of hidden units.")
    parser.add_argument("--num_graph_classes", type=int, help="Number of classes for the graph_classifier.",
    required=False)

    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
    )

    parser.add_argument(
        "--test_frac", type=float, default=0.20, help="Fraction of examples to be used as test split."
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.05, help="Fraction of examples to be used as val split."
    )
    parser.add_argument(
        "--pooling_op", type=str, default="mean", help="Pooling operation for the graph classification task.")

    return parser.parse_args()
