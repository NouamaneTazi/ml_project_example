""" 
searches for best params among all models in src/model_dispatcher
and saves results to logs/results_all_models.csv
* preprocessing parameters are defined in this file
* model parameters are defined in src/model_dispatcher.py
"""

import pandas as pd
import json
import itertools
from tqdm import tqdm
from src import config, models, search_best_params
from run_train import fill_df
from codecarbon import EmissionsTracker

preprocess_params = {
    "missing": [
        "mean",
        "median",
        "most_frequent",
        "remove_rows",
        "regression",
        "stochastic",
        "knn"],
    "scaling": [
        "standard",
        "min_max"],
    "add_Solids_log": [
        True,
        False],
    "poly_degree": [
        1,
        2,
        3]}

if __name__ == "__main__":
    with EmissionsTracker() as tracker:
        column_names = [
            "model_name",
            "preprocessing_params",
            "model_params",
            "accuracy",
            "auc",
            "f1_score"]
        df_results = pd.DataFrame(columns=column_names)
        path_save_results = f"./{config.SAVE_RESULTS}"

        # We must generate all the permutations
        params_names, values = zip(*preprocess_params.items())
        permutations_params = [dict(zip(params_names, v))
                            for v in itertools.product(*values)]
        for model_name, model_params in models.items():
            print(f"\\Searching best params for model: {model_name}")
            for preprocess_params in tqdm(permutations_params):
                # search best model params using `preprocess_params`
                metrics = search_best_params(
                    fold=-1,
                    train_data_path=f"./{config.TRAINING_FILE}",
                    model_name=model_name,
                    preprocess_params=preprocess_params)
                new_results = metrics.filter(
                    ["params", "mean_test_accuracy", "mean_test_roc_auc", "mean_test_f1"])
                new_results = new_results.rename(
                    columns={
                        "params": "model_params",
                        "mean_test_accuracy": "accuracy",
                        "mean_test_roc_auc": "auc",
                        "mean_test_f1": "f1_score"})
                new_results["model_params"] = new_results["model_params"].apply(
                    json.dumps)
                new_results[["model_name", "preprocessing_params"]] = [
                    model_name, json.dumps(preprocess_params)]
                df_results = df_results.append(new_results, ignore_index=True)
            print(f"Saving results to {path_save_results}...")
            try:
                df_before = pd.read_csv(path_save_results)
                df_tot = fill_df(df_before, df_results)
                df_tot.to_csv(path_save_results, index=False)
            except BaseException:
                df_results.to_csv(path_save_results, index=False)
            print("Results saved!\n")
