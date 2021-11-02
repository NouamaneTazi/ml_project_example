from sklearn import tree, ensemble, dummy, linear_model, svm, calibration
import xgboost as xgb

# TODO: hyperparameter tuning for each model.
models = {
    "constant": {
        "model": dummy.DummyClassifier,
        "base_model_params": {"constant": 0},
        "preprocessing_params": {"missing":"median", "scaling":"standard", "add_Solids_log":False, "poly_degree":1}
    },
    "dt_gini": {
        "model": tree.DecisionTreeClassifier,
        "base_model_params": {"criterion": "gini", "random_state": 42},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "dt_entropy": {
        "model":  tree.DecisionTreeClassifier,
        "base_model_params": {"criterion": "entropy"},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "rf": {
        "model": ensemble.RandomForestClassifier,
        "base_model_params": {"n_jobs": -1, "random_state": 42},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "xgb": {
        "model":  xgb.XGBClassifier,
        "base_model_params": {"n_jobs": -1, "random_state": 42},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "extratrees": {
        "model":  ensemble.ExtraTreesClassifier,
        "base_model_params": {"n_jobs": -1, "random_state": 42},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "log_reg": {
        "model":  linear_model.LogisticRegression,
        "base_model_params": {"n_jobs": -1, "random_state": 42},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "svm": {
        "model":lambda calib_params, svc_params: calibration.CalibratedClassifierCV(svm.LinearSVC(**svc_params), **calib_params),
        "base_model_params": {"svc_params":{"max_iter": 2000, "random_state": 42}, "calib_params": {"n_jobs": -1}},
        "preprocessing_params": {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    }

}
