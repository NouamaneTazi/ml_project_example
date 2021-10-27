# Drinking Water Potability

Kaggle: https://www.kaggle.com/artimule/drinking-water-probability

## Usage
* To train a model:
```bash
python3 src/train.py --fold 0 --model extratrees
```
* To predict on new instances / evaluate a model:
```bash
python src/inference.py \
--model extratrees \
--data input/drinking_water_potability.csv
```