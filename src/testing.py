import config
import joblib
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
from utils import save_logs


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
    predicted_classes = np.round(pred_probs)

    met = {}
    if y_test is not None:
        # get roc auc and accuracy and f1 score
        met["accuracy"] = metrics.accuracy_score(y_test, predicted_classes)
        met["auc"] = metrics.roc_auc_score(y_test, pred_probs)
        met["f1_score"] = metrics.f1_score(y_test, predicted_classes)

    return {"predict_prob": pred_probs, "metrics": met}