import config
import joblib
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
from utils import save_logs


def predict(test_data_path: str, model_name: str, model_path: str):
    # read testing data
    df = pd.read_csv(test_data_path)

    x_test = df.drop(["Potability", "kfold"], axis=1, errors='ignore').values
    y_test = df.Potability.values if "Potability" in df else None

    for fold in range(5):

        # fetch preprocessing pipeline
        pre_pipeline = joblib.load(
            Path(f"{model_path}/{model_name}/{model_name}_{fold}_preprocess.pkl"))
        # preprocess data
        x_test = pre_pipeline.transform(x_test)

        # fetch model
        clf = joblib.load(Path(
            f"{model_path}/{model_name}/{model_name}_{fold}.bin"))

        # predict
        # preds = clf.predict(x_test)
        pred_probs = clf.predict_proba(x_test)[:, 1]

        if fold == 0:
            predictions = pred_probs
        else:
            predictions += pred_probs

    # average over all folds
    predictions /= config.NUMBER_FOLDS
    predicted_classes = np.round(predictions)

    if y_test is not None:
        # get roc auc and accuracy and f1 score
        accuracy = metrics.accuracy_score(y_test, predicted_classes)
        auc = metrics.roc_auc_score(y_test, predictions)
        f1_score = metrics.f1_score(y_test, predicted_classes)
        print(
            f"Fold={-1}, Accuracy={accuracy}, F1-score={f1_score}, AUC={auc}")
        # save logs
        save_logs(model_name, -1, accuracy, f1_score, auc)

    df['Predictions'] = predictions
    df['Predicted Potability'] = predicted_classes
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model on provided data')
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
    # save initial test data with new Predictions column
    submission = predict(test_data_path=args.data,
                         model_name=args.model,
                         model_path=config.SAVED_MODELS)
    submission.to_csv(
        f"{config.PREDICTIONS}/{args.model}_submission.csv", index=False)
    print()
