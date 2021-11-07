# Drinking Water Potability

[![Pylint](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/pylint.yml/badge.svg)](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/pylint.yml)
[![Sync to Hugging Face hub](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/huggingfacehub.yml/badge.svg?branch=spaces)](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/huggingfacehub.yml)

Kaggle: <https://www.kaggle.com/artimule/drinking-water-probability>

## Usage

### Option 1

To train and evaluate a model

```bash
./run.sh extratrees
```

### Option 2

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
