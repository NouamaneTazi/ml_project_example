from . import config
import joblib
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics


def test(x_test, y_test, model):
    """
    Test :model:.
    :param x_test: the numpy array with test data
    :param y_test: the numpy array with test labels
    :param model: the model to test
    :return: dict with predicted probabilities and metrics (accuracy, AUC, f1 score)
    """
    # predict
    pred_probs = model.predict_proba(x_test)[:, 1]

    met = {}
    if y_test is not None:
        met = calculate_metrics(y_test, pred_probs)

    return {"predict_prob": pred_probs, "metrics": met}


def calculate_metrics(y_true, pred_probs):
    """ calculate metrics (accuracy, AUC, f1 score)
    :param y_true:
    :param pred_probs: of shape ()
    """
    met = {}
    predicted_classes = np.round(pred_probs)

    # get roc auc and accuracy and f1 score
    met["accuracy"] = metrics.accuracy_score(y_true, predicted_classes)
    met["f1_score"] = metrics.f1_score(y_true, predicted_classes)
    met["auc"] = metrics.roc_auc_score(y_true, pred_probs)
    return met
