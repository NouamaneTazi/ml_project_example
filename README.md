# ML Example Project

[![Pylint](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/pylint.yml/badge.svg)](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/pylint.yml)
[![Sync to Hugging Face hub](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/huggingfacehub.yml/badge.svg?branch=spaces)](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/huggingfacehub.yml)

This project was made as a good _production ready_ example for a classical machine learning project. We tried to incorporate as many good ml habits and guidelines from the book _Approaching (Almost) Any Machine Learning Problem_ by _Abhishek Thakur_ and others.

The goal is to classify potable water using the data provided in here: <https://www.kaggle.com/artimule/drinking-water-probability>

### Stacking  

[notebook](https://nbviewer.org/github/NouamaneTazi/drinking_water_potability/blob/master/notebooks/Ensembling.ipynb)

![https://towardsdatascience.com/ensemble-learning-stacking-blending-voting-b37737c4f483](https://miro.medium.com/max/1250/1*CoauXirckomVXxw2Id2w_Q.jpeg)
The best model was trained using the **Stacking Generalization** method where we first train a couple of weak learners, then use the predictions from these models to train a **Meta-Model** able to reduce the generalization error of the different weak learners and provide more robust predictions.

### Interpretability

 [notebook](https://nbviewer.org/github/NouamaneTazi/drinking_water_potability/blob/master/notebooks/Interpret%20results.ipynb)

![individual prediction explanation](https://user-images.githubusercontent.com/29777165/142934130-85c37332-a668-46d5-b889-b13664f098d3.png)

We used the **SHAP** (SHapley Additive exPlanations) method to explain the model's predictions. You can find a great explanation for this method in the [interpretable ML book](https://christophm.github.io/interpretable-ml-book/shap.html).

### HuggingFace Space - Gradio

A [demo app](https://huggingface.co/spaces/nouamanetazi/drinking_water_potability) was made using Gradio, and published to a HuggingFace space, to enable users to play around with parameters and test the consistency of the model.

### CI/CD

We also set up a [Github workflow](https://github.com/NouamaneTazi/drinking_water_potability/actions/workflows/huggingfacehub.yml) to automatically deploy the best model to the HuggingFace Space.

## Usage

### Initialization

Before running the models, you should put your data file (`drinking_water_potability.csv`) in the _input/raw_ folder.

#### Option 1

To train and evaluate a model

```bash
./run.sh extratrees
```

#### Option 2

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
