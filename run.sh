#!/bin/sh
# -W ignore : ignore all warnings
echo "Training on all folds (showing OOF scores)" && \
python3 -W ignore src/train.py --fold -1 --model $1 && \
echo "Evaluating trained model on whole dataset" && \
python3 -W ignore src/inference.py --model $1