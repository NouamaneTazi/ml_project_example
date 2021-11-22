from typing import Any
import joblib
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics

from . import config
from .utils import save_logs, CustomUnpickler
from .evaluate import test


def predict_one_sample(sample: np.array,
                       model_name: str,
                       model_path: str = config.SAVED_MODELS,
                       fold: int = -1) -> tuple[np.array,
                                                np.array,
                                                Any,
                                                Any]:
    """
    Predict water drinking potability class and probablities from one sample data
    :param sample: array-like of length 9 containing data used for prediction
    :param model_name: the model to use for prediction (must be stored in `model_path`)
    :param model_path: the path of the pretrained model
    :return: a tuple containing water drinking potability and classes probabilities
    """
    if not isinstance(sample, (list, tuple, np.ndarray)):
        raise TypeError("sample must be array like")
    if len(sample) != 9:
        raise ValueError("Provided sample must be of length: 9")

    x_test = np.array(sample).reshape(1, -1)

    # fetch preprocessing pipeline and model
    try:
        pre_pipeline = joblib.load(
            Path(f"{model_path}/{model_name}/{model_name}_{fold}_preprocess.pkl"))
        clf = joblib.load(
            Path(f"{model_path}/{model_name}/{model_name}_{fold}.bin"))
    except BaseException:
        with open(Path(f"{model_path}/{model_name}/{model_name}_{fold}_preprocess.pkl"), 'rb') as f:
            pre_pipeline = CustomUnpickler(f).load()
        with open(Path(f"{model_path}/{model_name}/{model_name}_{fold}.bin"), 'rb') as f:
            clf = CustomUnpickler(f).load()

    # preprocess data
    x_test_processed, _ = pre_pipeline.transform(x_test)
    # predict
    pred_probs = clf.predict_proba(x_test_processed)[0]
    predicted_classes = np.argmax(pred_probs)

    return predicted_classes, pred_probs, pre_pipeline, clf


def _predict(fold: int, test_data_path: str, model_name: str, model_path: str):
    """
    Generate metrics on test dataset for model :model_path:.
    :param fold: the fold id used for validation
    :param test_data_path: the path for the test dataframe
    :param model_name: the model configuration
    :param model_path: the path of the pretrained model
    :return: a dict containing the predicted labels and the corresponding metrics (accuracy, AUC, f1 score)
    """
    df = pd.read_csv(test_data_path)

    x_test = df.drop(["Potability", "kfold"], axis=1, errors='ignore').values
    y_test = df.Potability.values if "Potability" in df else None

    if fold == -1:
        for fold_k in range(config.NUMBER_FOLDS):
            results_test = _predict(
                fold_k, test_data_path, model_name, model_path)
            if fold_k == 0:
                pred_prob = results_test["predict_prob"]
            else:
                pred_prob += results_test["predict_prob"]
        # average over all folds
        pred_prob /= config.NUMBER_FOLDS
        predicted_classes = np.round(pred_prob)

        met = {}
        if y_test is not None:
            # get roc auc and accuracy and f1 score
            met["accuracy"] = metrics.accuracy_score(y_test, predicted_classes)
            met["auc"] = metrics.roc_auc_score(y_test, pred_prob)
            met["f1_score"] = metrics.f1_score(y_test, predicted_classes)
            print(
                f"Fold=-1, Accuracy={met['accuracy']}, F1-score={met['f1_score']}, AUC={met['auc']}")
            # save logs
            save_logs(model_name, -
                      1, met["accuracy"], met["f1_score"], met["auc"])
            df['Predictions'] = pred_prob
            df['Predicted Potability'] = predicted_classes
            df.to_csv(
                f"{config.PREDICTIONS}/{args.model}_submission.csv",
                index=False)
        return {"predict_prob": pred_prob, "metrics": met}

    # fetch preprocessing pipeline
    pre_pipeline = joblib.load(
        Path(f"{model_path}/{model_name}/{model_name}_{fold}_preprocess.pkl"))
    # preprocess data
    x_test_processed = pre_pipeline.transform(x_test)

    # fetch model
    clf = joblib.load(
        Path(f"{model_path}/{model_name}/{model_name}_{fold}.bin"))

    # predict
    results = test(x_test_processed, y_test, clf)
    met = results["metrics"]
    if y_test is not None:
        print(
            f"Fold={fold}, Accuracy={met['accuracy']}, F1-score={met['f1_score']}, AUC={met['auc']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate a model on provided data')
    parser.add_argument(
        "--fold",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stacking"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.TESTING_FILE
    )
    args = parser.parse_args()
    print()
    # save initial test data with new Predictions column
    _predict(fold=args.fold,
             test_data_path=args.data,
             model_name=args.model,
             model_path=config.SAVED_MODELS)
    print()
