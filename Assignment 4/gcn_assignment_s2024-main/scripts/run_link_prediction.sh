#!/bin/bash
# Runs link prediction with the default parameters. Tested with "cora" or "citeseer" datasets.
GRAPH="$1"
python src/training/train_link_prediction.py \
  --graph "${GRAPH}" \
  --task link_pred
