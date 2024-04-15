#!/bin/bash
# Runs link prediction with the default parameters. Tested with "cora" or "citeseer" datasets.
# GRAPH="karate"
# python -m src.training.train_link_prediction --graph "${GRAPH}" --task link_pred
GRAPH="citeseer"
python -m src.training.train_link_prediction --graph "${GRAPH}" --task link_pred