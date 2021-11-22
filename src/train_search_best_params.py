from . import config, model_dispatcher
import pandas as pd
from sklearn.model_selection import GridSearchCV
from .utils import save_file

from .preprocessing import preprocessing_pipeline


def preprocess(
        X,
        y,
        model_name,
        fold=None,
        preprocess_params=None,
        save=True,
        return_pipeline=False):
    """
    This function is used for feature engineering
    :param X: the numpy array with data
    :param model_name: the model name
    :param fold: the fold id used for validation
    :param preprocess_params: (optional) parameters for the preprocessing step
    :param save: (optional) whether to save trained pipeline
    :param return_pipeline: (optional) whether to return trained pipeline
    :return: preprocessed X and y (and optionally preprocessing pipeline)
    """
    # fetch preprocessing params from model_dispatcher
    preprocessing_params = preprocess_params or model_dispatcher.models[model_name].get(
        "preprocessing_params", {})
    pre_pipeline = preprocessing_pipeline(**preprocessing_params)

    # preprocess data
    X, y = pre_pipeline.fit_transform(X, y)

    if save:
        save_file(
            pre_pipeline,
            f"{config.SAVED_MODELS}/{model_name}/{model_name}_{fold}_preprocess_search_params.pkl")

    if return_pipeline:
        return X, y, pre_pipeline
    else:
        return X, y


def search_best_params(
        fold: int,
        train_data_path: str,
        model_name: str,
        preprocess_params: dict = None,
        base_model_params: dict = None,
        search_model_params: dict = None):
    """
    Search for best params for model :model_name: with a GridSearch, on the indicated fold.
    :param fold: the fold id used for training (-1 for all)
    :param train_data_path: the path for the train dataframe
    :param model_name: the model configuration
    :param preprocess_params: (optional) parameters for the preprocessing step
    :param base_model_params: (optional) the base model configuration
    :search_model_params: (optional) the different model configurations to test
    :return: DataFrame of models with metrics (accuracy, AUC, f1 score)
    """
    df = pd.read_csv(train_data_path)
    if fold != -1:
        df = df[df["kfold"] == fold].reset_index(drop=True)
    X = df.drop(["Potability", "kfold"], axis=1).values
    y = df.Potability.values
    X, y = preprocess(X, y, model_name, fold, preprocess_params)

    # Instantiate GridSearch
    clf = model_dispatcher.models[model_name]["model"](
        **model_dispatcher.models[model_name]["base_model_params"])
    if base_model_params:
        clf.set_params(**base_model_params)
    search_model_params = search_model_params if search_model_params else model_dispatcher.models[
        model_name]["search_model_params"]
    gs = GridSearchCV(
        clf,
        search_model_params,
        scoring=(
            'accuracy',
            'roc_auc',
            'f1'),
        refit=False,
        n_jobs=-1,
        verbose=0)

    # Train and output
    gs.fit(X, y)
    df_results = pd.DataFrame(gs.cv_results_)
    df.to_csv(
        f"{config.SEARCH_PARAMS_LOGS_FOLDER}/{model_name}_{fold}.csv",
        index=False)
    return df_results
