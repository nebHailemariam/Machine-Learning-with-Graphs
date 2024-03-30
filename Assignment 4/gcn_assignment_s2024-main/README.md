#### This is an identical copy of [Aman's design of this homework](https://github.com/madaan/gcn_assignment)

# GCN Assignment for 11441/11741 (Spring 2024)

## Getting started

- Unzip the data from `data.zip.`
- Install the required packages listed in `requirements.txt.`
- Set [PYTHONPATH](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html) to `.:../:src` (you can do this by running `export PYTHONPATH=".:../:src"` on the shell, for example).
- Please use Python 3.6 or above.

## Outline

- The goal of this assignment is to implement a basic GCN model for node classification, link prediction, and graph classification. Most of the code related to training, data loading, and evaluation is provided. You need to implement the parts marked with a #TODO. Each TODO needs a few lines of code to complete (2-3 lines in most cases).

### Part 1: Graph exploration

- The notebook [notebooks/GraphExploration.ipynb](notebooks/GraphExploration.ipynb) contains the code for the first part of this assignment. This part aims to introduce you to [graph.py](src/data/graph.py), the class that represents a graph and some of its methods.

### Part 2: Implementing a GCN

- This has two sub-parts:

1. Implementing a GCN layer by completing [this code](src/modeling/core/gcn.py).
2. Using the GCN layer to complete the implementation of GCN [here](src/modeling/core/layers.py).

### Part 3: Node classification

- Use the GCN model implemented in part 2 to classify nodes in the graph (by completing [this](src/modeling/tasks/node_classification.py)).

### Part 4: Link prediction

- Use the GCN model implemented in part 2 to predict links in the graph ([partial implementation](src/modeling/tasks/link_prediction.py)). This [notebook](notebooks/LinkPredictionTrainingData.ipynb) can be used to explore the training data.

### Part 5: Graph classification

- We will use the GCN model implemented in part 2 to classify graphs (i.e., predict a label for each graph, partial implementation [here](src/modeling/tasks/graph_classification.py),  [notebook](notebooks/GraphClassificationStatistics.ipynb)).

## Scripts

### Node classification

```sh
bash scripts/run_node_classification.sh [GRAPH_NAME]
```

Where:

- GRAPH_NAME is the name of the graph to use (e.g., `cora,` `citeseer`).

E.g., to run node classification on `citeseer` with topological features, run:

```sh
bash scripts/run_node_classification.sh citeseer_plus_topo
```

- The usage link prediction is similar.

### Graph classification

```sh
bash scripts/run_graph_classification.sh [GRAPH_PATH] [NUM_CLASSES] [POOLING_OP (max|mean|last)] [NUM_EPOCHS]
```

Where:

- GRAPH_PATH: path to the graph file.
- NUM_CLASSES: number of classes in the graph.
- POOLING_OP: pooling operation to use.
- NUM_EPOCHS: number of epochs to train the model.

More details can be found in scripts/run_graph_classification.sh.

E.g., to run graph classification on mutag (2 classes) with max-pooling and 200 epochs:

```sh
bash scripts/run_graph_classification.sh data/graph_classification/graph_mutag_with_node_num.pt 2 max 200
```

## FAQ

### Getting feedback

- You can create a **private** fork of this repository on GitHub and add the TAs as collaborators (usernames: Edward-Sun). This might help you in asking questions without having to copy-paste your code on piazza (you can just reference Github code/copy permalink). You can use [these instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository) or just copy-paste the code into a new repository.

### Posting your solutions online

- As with all the other assignments, please do not share your solutions publicly.

## Attributions

- Parts of the data loader code are based on [this repo](https://github.com/tkipf/gcn).
