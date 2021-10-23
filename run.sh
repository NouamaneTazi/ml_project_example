#!/bin/sh
python src/train.py --fold 0
python src/train.py --fold 1
python src/train.py --fold 2
python src/train.py --fold 3
python src/train.py --fold 4