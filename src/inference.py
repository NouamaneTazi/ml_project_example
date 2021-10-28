import config
import joblib
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
from utils import save_logs

from testing import test


def predict(fold: int, test_data_path: str, model_name: str, model_path: str):
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
            results_test = predict(fold_k, test_data_path, model_name, model_path)
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
            print(f"Fold=-1, Accuracy={met['accuracy']}, F1-score={met['f1_score']}, AUC={met['auc']}")
            # save logs
            save_logs(model_name, -1, met["accuracy"], met["f1_score"], met["auc"])
            df['Predictions'] = pred_prob
            df['Predicted Potability'] = predicted_classes
            df.to_csv(f"{config.PREDICTIONS}/{args.model}_submission.csv", index=False)
        return {"predict_prob": pred_prob, "metrics": met}

    # fetch preprocessing pipeline
    pre_pipeline = joblib.load(
        Path(f"{model_path}/{model_name}/{model_name}_{fold}_preprocess.pkl"))
    # preprocess data
    x_test_processed = pre_pipeline.transform(x_test)

    # fetch model
    clf = joblib.load(Path(f"{model_path}/{model_name}/{model_name}_{fold}.bin"))

    # predict
    results = test(x_test_processed, y_test, clf)
    met = results["metrics"]
    if y_test is not None:
        print(f"Fold={fold}, Accuracy={met['accuracy']}, F1-score={met['f1_score']}, AUC={met['auc']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model on provided data')
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
        default=config.TESTING_FILE
    )
    args = parser.parse_args()
    print()
    # save initial test data with new Predictions column
    predict(fold=args.fold,
            test_data_path=args.data,
            model_name=args.model,
            model_path=config.SAVED_MODELS)
    print()
