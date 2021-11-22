import os
from . import config, model_dispatcher
import time
import joblib
import pickle
import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.impute import SimpleImputer
from .preprocessing import preprocessing_pipeline
from .model_dispatcher import StackingClf, StackingPrePipeline, Learner


def save_file(object, path):
    """ saves object to path """
    # create corresponding folder if doesnt exist
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # save object
    joblib.dump(object, path)


def save_logs(model_name, fold, accuracy, f1_score, auc):
    """saves prediction logs to `config.TRAIN_LOGS_FILE` (`logs/scores.csv` by default)"""
    preprocessing_params = model_dispatcher.models[model_name].get(
        "preprocessing_params", {})
    with open(config.TRAIN_LOGS_FILE, 'a') as f:
        csv.writer(f).writerow(
            [time.strftime('%y%m%d-%H%M%S'),
             model_name,
             fold,
             json.dumps(preprocessing_params),
             round(accuracy, 5),
             round(f1_score, 5),
             round(auc, 5)])


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)
