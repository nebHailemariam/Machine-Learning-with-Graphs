#!/bin/bash
# Runs node classification on the given graph. 

# To run on MUTAG, set
#   GRAPH: data/graph_classification/graph_mutag_with_node_num.pt
#   CLASSES: 2 classes
#   EPOCHS: 200

# To run on ENZYMES, set
#   GRAPH: data/graph_classification/graph_enzymes_with_node_num.pt
#   CLASSES: 6 classes
#   EPOCHS: 1000

# The pooling op can be `mean`, `max`, or `last`.

GRAPH="$1"
NUM_CLASSES="$2"
POOLING_OP="$3"
NUM_EPOCHS="$4"
python src/training/train_graph_classification.py \
  --task graph_classification \
  --graphs_path ${GRAPH} \
  --pooling_op ${POOLING_OP} \
  --num_graph_classes "${NUM_CLASSES}" \
  --lr 1e-3 \
  --epochs ${NUM_EPOCHS}
