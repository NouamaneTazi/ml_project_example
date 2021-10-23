import os
import config
import model_dispatcher
import time
import joblib
import argparse

import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.impute import SimpleImputer
import preprocessing


def preprocess(x_train, x_valid, model):
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param model: the model configuration
    :return: dataframe with new features
    """
    pre_pipeline = preprocessing.preprocessing_pipeline()

    # drops kfold column because it's not used in preprocessing
    kfold_col = df['kfold']
    df = df.drop(columns="kfold")

    initial_cols = list(df.columns)
    # TODO: find added columnns automatically from pipeline
    added_cols = ['Solids_log']
    # Preprocess data (returns numpy array)
    arr = pre_pipeline.fit_transform(df)

    df = pd.DataFrame(arr, columns=initial_cols + added_cols)
    df['kfold'] = kfold_col
    return df


def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # handle missing data
    # fill all missing data with -1 and let decision tree handle it
    imp_constant = SimpleImputer(strategy='constant', fill_value=-1)
    df = pd.DataFrame(imp_constant.fit_transform(df), columns=df.columns)

    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the Potability column from dataframe and convert it to
    # a numpy array by using .values.
    # target is Potability column in the dataframe
    x_train = df_train.drop("Potability", axis=1).values
    y_train = df_train.Potability.values
    # similarly, for validation, we have
    x_valid = df_valid.drop("Potability", axis=1).values
    y_valid = df_valid.Potability.values

    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    valid_preds = clf.predict(x_valid)
    valid_probs = clf.predict_proba(x_valid)
    # get roc auc and accuracy score
    accuracy = metrics.accuracy_score(y_valid, valid_preds)
    auc = metrics.roc_auc_score(
        df_valid.Potability.values, valid_probs[:, 1])
    print(f"Fold={fold}, Accuracy={accuracy}, AUC={auc}")
    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT,
                f"{model}_{fold}__{round(auc,3)}_{time.strftime('%m%d-%H%M%S')}.bin"))


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        default=0
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf"
    )
    args = parser.parse_args()
    run(fold=args.fold, model=args.model)
