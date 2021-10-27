import os
import config
import model_dispatcher
import time
import joblib
import argparse
from pathlib import Path

import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.impute import SimpleImputer
from preprocessing import preprocessing_pipeline


def preprocess(x_train, x_valid, model_name, fold):
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param model: the model configuration
    :return: dataframe with new features
    """
    # fetch preprocessing params from model_dispatcher
    preprocessing_params = model_dispatcher.models[model_name].get(
        "preprocessing_params", {})
    pre_pipeline = preprocessing_pipeline(**preprocessing_params)

    # preprocess data
    x_train = pre_pipeline.fit_transform(x_train)
    x_valid = pre_pipeline.transform(x_valid)

    # create corresponding folder if doesnt exist
    Path(f"{config.SAVED_MODELS}/{model_name}")\
        .mkdir(parents=True, exist_ok=True)
    # save model
    joblib.dump(
        pre_pipeline, Path(f"{config.SAVED_MODELS}/{model_name}/{model_name}_{fold}_preprocess.pkl"))

    return x_train, x_valid


def run(fold: int, model_name: str):
    """
    Fits :model_name: on the rest of folds, and validates on the fold :fold:.
    It also saves the trained model.
    :param fold: the fold id used for validation
    :param model_name: the model configuration
    """
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the Potability column from dataframe and convert it to
    # a numpy array.
    # target is Potability column in the dataframe
    x_train = df_train.drop(["Potability", "kfold"], axis=1).values
    y_train = df_train.Potability.values
    # similarly, for validation, we have
    x_valid = df_valid.drop(["Potability", "kfold"], axis=1).values
    y_valid = df_valid.Potability.values

    # Preprocess data
    x_train, x_valid = preprocess(x_train, x_valid, model_name, fold)

    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model_name]["model"]

    # fit the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    valid_preds = clf.predict(x_valid)
    valid_probs = clf.predict_proba(x_valid)
    # get roc auc and accuracy score
    accuracy = metrics.accuracy_score(y_valid, valid_preds)
    auc = metrics.roc_auc_score(
        df_valid.Potability.values, valid_probs[:, 1])
    f1_score = metrics.f1_score(y_valid, valid_preds)
    print(f"Fold={fold}, Accuracy={accuracy}, F1-score={f1_score}, AUC={auc}")
    # save the model
    # joblib.dump(clf, os.path.join(config.SAVED_MODELS,
    #             f"{model_name}_{fold}__{round(auc,3)}_{time.strftime('%m%d-%H%M%S')}.bin"))

    # create corresponding folder if doesnt exist
    Path(f"{config.SAVED_MODELS}/{model_name}")\
        .mkdir(parents=True, exist_ok=True)
    # save model
    joblib.dump(
        clf, Path(f"{config.SAVED_MODELS}/{model_name}/{model_name}_{fold}.bin"))


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf"
    )
    args = parser.parse_args()
    print()
    if args.fold == -1:
        for fold in range(config.NUMBER_FOLDS):
            run(fold=fold, model_name=args.model)
    else:
        run(fold=args.fold, model_name=args.model)
    print()
