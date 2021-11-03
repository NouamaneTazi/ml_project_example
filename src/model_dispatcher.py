from sklearn import tree, ensemble, dummy, linear_model, svm, calibration, naive_bayes, neighbors
import xgboost as xgb

# TODO: hyperparameter tuning for each model.
models = {
    "constant": {
        "model": dummy.DummyClassifier,
        "base_model_params": {"constant": 0, "random_state": 42},
        "search_model_params": {"strategy": ["stratified", "most_frequent", "prior", "uniform", "constant"]},
        "preprocessing_params": {"missing":"median", "scaling":"standard", "add_Solids_log":False, "poly_degree":1}
    },
    "dt_gini": {
        "model": tree.DecisionTreeClassifier,
        "base_model_params": {"criterion": "gini", "random_state": 42},
        "search_model_params": {"splitter": ["best", "random"], "class_weight": [None, "balanced"]},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "dt_entropy": {
        "model":  tree.DecisionTreeClassifier,
        "base_model_params": {"criterion": "entropy"},
        "search_model_params": {"splitter": ["best", "random"], "class_weight": [None, "balanced"]},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "rf": {
        "model": ensemble.RandomForestClassifier,
        "base_model_params": {"n_jobs": -1, "random_state": 42},
        "search_model_params": {"criterion": ["gini", "entropy"], "class_weight": [None, "balanced"]},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "xgb": {
        "model":  xgb.XGBClassifier,
        "base_model_params": {"n_jobs": -1, "random_state": 42},
        "search_model_params": {"lambda": [0, 0.5, 1], "alpha": [0, 0.5, 1]},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "extratrees": {
        "model":  ensemble.ExtraTreesClassifier,
        "base_model_params": {"n_jobs": -1, "random_state": 42},
        "search_model_params": {"criterion": ["gini", "entropy"], "class_weight": [None, "balanced"]},
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "log_reg": {
        "model":  linear_model.LogisticRegression,
        "base_model_params": {"n_jobs": -1, "random_state": 42},
        "search_model_params": [{"solver": ["lbfgs"], "penalty": ["none", "l2"]}, {"solver": ["liblinear"], "penalty": ["l1", "l2"]}],
        "preprocessing_params":  {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "svm_calib": {
        "model":lambda calib_params, svc_params: calibration.CalibratedClassifierCV(svm.LinearSVC(**svc_params), **calib_params),
        "base_model_params": {"svc_params":{"max_iter": 2000, "random_state": 42}, "calib_params": {"n_jobs": -1}},
        "search_model_params": {"ensemble": [True, False], "method": ["sigmoid", "isotonic"]},
        "preprocessing_params": {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "svm": {
        "model":svm.SVC,
        "base_model_params": {"max_iter": 2000, "random_state": 42},
        "search_model_params": {"kernel": ["linear", "poly", "rbf", "sigmoid"], "class_weight": [None, "balanced"]},
        "preprocessing_params": {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "naive_bayes": {
        "model":naive_bayes.GaussianNB,
        "base_model_params": {},
        "search_model_params": {"var_smoothing": [1e-10, 1e-9, 1e-8, 1e-5]},
        "preprocessing_params": {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    },
    "knn": {
        "model":neighbors.KNeighborsClassifier,
        "base_model_params": {"n_jobs": -1},
        "search_model_params": {"p": [1, 2, 3], "weights": ["uniform", "distance"], "n_neighbors": [3, 5, 8, 12]},
        "preprocessing_params": {"missing": "median", "scaling": "standard", "add_Solids_log": True, "poly_degree": 1}
    }
}
