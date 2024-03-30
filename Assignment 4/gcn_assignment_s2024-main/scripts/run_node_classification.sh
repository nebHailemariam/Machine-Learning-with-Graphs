#!/bin/bash
# Runs node classification on the given graph. Tested on cora and citeseer.
GRAPH="$1"
python3 src/training/train_node_classification.py --graph "${GRAPH}"  --task classify --epochs 300
