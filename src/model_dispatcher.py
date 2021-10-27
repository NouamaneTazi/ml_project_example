from sklearn import tree, ensemble
import xgboost as xgb

# TODO: hyperparameter tuning for each model.
models = {
    "dt_gini": {
        "model": tree.DecisionTreeClassifier(
            criterion="gini",
            random_state=42
        ),
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "dt_entropy": {
        "model":  tree.DecisionTreeClassifier(
            criterion="entropy",
            random_state=42,
        ),
    },
    "rf": {
        "model": ensemble.RandomForestClassifier(n_jobs=-1,
                                                 random_state=42),
        "preprocessing_params":  {"add_Solids_log": False}
    },
    "xgb": {
        "model":  xgb.XGBClassifier(
            n_jobs=-1,
            random_state=42
        ),
        "preprocessing_params":  {"add_Solids_log": False}
    },
}
