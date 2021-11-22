"""
trains the models that are defined in src/model_dispatcher.py
and saves results to logs/results_all_models.csv
"""

import pandas as pd
import json

from src import config, models, train


def fill_df(df_main, df_new):
    for _, new_row in df_new.iterrows():
        to_add = True
        for _, row in df_main.iterrows():
            if row["model_name"] == new_row["model_name"] and json.loads(
                row["preprocessing_params"]) == json.loads(
                new_row["preprocessing_params"]) and json.loads(
                row["model_params"]) == json.loads(
                    new_row["model_params"]):
                to_add = False
        if to_add:
            df_main = df_main.append(new_row)
    return df_main


if __name__ == "__main__":
    column_names = [
        "model_name",
        "preprocessing_params",
        "model_params",
        "accuracy",
        "auc",
        "f1_score"]
    df_results = pd.DataFrame(columns=column_names)
    for model_name, model_params in models.items():
        metrics = train(
            fold=-1,
            train_data_path=f"./{config.TRAINING_FILE}",
            model_name=model_name)
        new_result = {
            "model_name": model_name, "preprocessing_params": json.dumps(
                models[model_name]["preprocessing_params"]), "model_params": json.dumps(
                models[model_name]["base_model_params"]), **metrics}
        df_results = df_results.append(new_result, ignore_index=True)
    print("Saving results...")
    path_save_results = f"./{config.SAVE_RESULTS}"
    try:
        df_before = pd.read_csv(path_save_results)
        df_tot = fill_df(df_before, df_results)
        df_tot.to_csv(path_save_results, index=False)
    except BaseException:
        df_results.to_csv(path_save_results, index=False)
    print("Results saved!")
