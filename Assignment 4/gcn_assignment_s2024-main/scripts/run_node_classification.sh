#!/bin/bash
# Runs node classification on the given graph. Tested on cora and citeseer.
GRAPH="cora"
python -m src.training.train_node_classification --graph "${GRAPH}" --task classify --epochs 300
 