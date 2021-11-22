from . import config, model_dispatcher
import argparse

import pandas as pd
from .preprocessing import preprocessing_pipeline
from .utils import save_file, save_logs
from sklearn import metrics


def preprocess(
        x_train,
        y_train,
        x_valid,
        y_valid,
        model_name=None,
        fold=None,
        preprocess_params=None,
        save=True):
    """
    This function is used for feature engineering
    :param x_train: the numpy array with train data
    :param x_valid: the numpy array with valid data
    :param model_name: the model name
    :param fold: the fold id used for validation
    :param preprocess_params: (optional) parameters for the preprocessing step
    :return: x_train and x_test, preprocessed
    """
    # fetch preprocessing params from model_dispatcher
    preprocessing_params = preprocess_params or model_dispatcher.models[model_name].get(
        "preprocessing_params", {})
    pre_pipeline = preprocessing_pipeline(**preprocessing_params)

    # preprocess data
    x_train, y_train = pre_pipeline.fit_transform(x_train, y_train)
    x_valid, y_valid = pre_pipeline.transform(x_valid, y_valid)

    if save:
        save_file(
            pre_pipeline,
            f"{config.SAVED_MODELS}/{model_name}/{model_name}_{fold}_preprocess.pkl")

    return x_train, y_train, x_valid, y_valid


def train_validate(x_train, y_train, x_valid, y_valid, model):
    """
    Fits :model: on :x_train:, and validates on :x_valid:.
    :param x_train: the numpy array with train data
    :param y_train: the numpy array with train labels
    :param x_valid: the numpy array with valid data
    :param y_valid: the numpy array with valid labels
    :param fold: the fold id used for validation
    :param model: the model (already built)
    :return: a dict containing the trained model and metrics (accuracy, AUC, f1 score)
    """
    # fit the model on training data
    model.fit(x_train, y_train)
    # create predictions for validation samples
    valid_preds = model.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, valid_preds)
    f1_score = metrics.f1_score(y_valid, valid_preds)
    try:
        valid_probs = model.predict_proba(x_valid)
        auc = metrics.roc_auc_score(y_valid, valid_probs[:, 1])
    except BaseException:
        auc = -1
    return {
        "model": model,
        "metrics": {
            "accuracy": accuracy,
            "auc": auc,
            "f1_score": f1_score}}


def train_save(
        x_train,
        y_train,
        x_valid,
        y_valid,
        fold: int,
        model_name: str,
        model_params: dict = None):
    """
    Train and test :model_name: and save it.
    :param x_train: the numpy array with train data
    :param y_train: the numpy array with train labels
    :param x_valid: the numpy array with valid data
    :param y_valid: the numpy array with valid labels
    :param fold: the fold id used for validation
    :param model_name: the model name
    :param model_params: (optional) the model configuration
    :return: metrics (accuracy, AUC, f1 score)
    """
    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model_name]["model"](
        **model_dispatcher.models[model_name]["base_model_params"])
    if model_params:
        clf.set_params(**model_params)

    results_training = train_validate(x_train, y_train, x_valid, y_valid, clf)
    metrics = results_training["metrics"]

    # save model
    save_file(
        clf, f"{config.SAVED_MODELS}/{model_name}/{model_name}_{fold}.bin")

    # save logs
    save_logs(
        model_name,
        fold,
        metrics['accuracy'],
        metrics['f1_score'],
        metrics['auc'])

    return metrics


def run(
        fold: int,
        train_data_path: str,
        model_name: str,
        preprocess_params: dict = None,
        model_params: dict = None):
    """
    Preprocess data and fits :model_name: on the rest of folds, and validates on the fold :fold:.
    :param fold: the fold id used for validation
    :param train_data_path: the path for the train dataframe
    :param model_name: the model configuration
    :param preprocess_params: (optional) parameters for the preprocessing step
    :param model_params: (optional) the model configuration
    :return: metrics (accuracy, AUC, f1 score)
    """
    if fold == -1:
        metrics = {"accuracy": 0, "auc": 0, "f1_score": 0}
        for fold_k in range(config.NUMBER_FOLDS):
            new_metrics = run(
                fold=fold_k,
                train_data_path=train_data_path,
                model_name=model_name)
            metrics = {k: v + new_metrics[k] for k, v in metrics.items()}
        # TODO: this is not necessarily correct
        metrics = {k: v / config.NUMBER_FOLDS for k, v in metrics.items()}
        print(
            f"Avg on all folds, Accuracy={metrics['accuracy']}, F1-score={metrics['f1_score']}, AUC={metrics['auc']}")
        return metrics

    # read the training data with folds
    df = pd.read_csv(train_data_path)

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
    x_train, y_train, x_valid, y_valid = preprocess(
        x_train, y_train, x_valid, y_valid, model_name, fold, preprocess_params)

    metrics = train_save(
        x_train,
        y_train,
        x_valid,
        y_valid,
        fold,
        model_name,
        model_params)
    print(
        f"Fold={fold}, Accuracy={metrics['accuracy']}, F1-score={metrics['f1_score']}, AUC={metrics['auc']}")
    return metrics


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser(
        'Train specified model on all fold but specified fold.')
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
    parser.add_argument(
        "--data",
        type=str,
        default=config.TRAINING_FILE
    )
    args = parser.parse_args()
    print()
    run(fold=args.fold, train_data_path=args.data, model_name=args.model)
    print()
