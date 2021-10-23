#!/bin/sh
python src/train.py --fold 0 --model rf 
python src/train.py --fold 1 --model rf
python src/train.py --fold 2 --model rf
python src/train.py --fold 3 --model rf
python src/train.py --fold 4 --model rf