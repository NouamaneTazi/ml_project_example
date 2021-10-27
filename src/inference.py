import config
import joblib
import argparse
from pathlib import Path
import pandas as pd


def predict(test_data_path: str, model_name: str, model_path: str):
    # read testing data
    df = pd.read_csv(test_data_path)

    # Preprocess data
    x_test = df.values

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

    df['Predictions'] = predictions
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="rf"
    )
    args = parser.parse_args()

    # save initial test data with new Predictions column
    submission = predict(test_data_path=config.TESTING_FILE,
                         model_name=args.model,
                         model_path=config.SAVED_MODELS)
    submission.to_csv(
        f"{config.PREDICTIONS}/{args.model}_submission.csv", index=False)
