import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


class ModifiedPreprocessingPipeline:
    def __init__(self, missing, pipeline):
        self.missing = missing
        self.pipeline = pipeline

    def fit_transform(self, X, y=None):
        if self.missing == "remove_rows":
            if isinstance(X, pd.DataFrame):
                X = X.values
            to_keep = ~np.isnan(X).any(axis=1)
            X = X[to_keep, :]
            if isinstance(y, np.ndarray):
                y = y[to_keep]
        return self.pipeline.fit_transform(X), y

    def transform(self, X, y=None):
        if self.missing == "remove_rows":
            if isinstance(X, pd.DataFrame):
                X = X.values
            to_keep = ~np.isnan(X).any(axis=1)
            X = X[to_keep, :]
            if isinstance(y, np.ndarray):
                y = y[to_keep]
        return self.pipeline.transform(X), y


class RemoveNull(BaseEstimator, TransformerMixin):
    '''Defines a transformer to delete rows or cols containing null values'''

    def __init__(self, direction=0):
        self.direction = direction

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.direction == 0:
            return X[:, ~np.isnan(X).any(axis=0)]
        else:
            return X


class SolidsLogAdder(BaseEstimator, TransformerMixin):
    '''Defines a transformer to add Solids log feature'''

    solids_ix = 2

    def __init__(self, add_Solids_log=True):
        self.add_Solids_log = add_Solids_log

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.add_Solids_log:
            Solids_log = np.log1p(X[:, self.solids_ix])
            return np.c_[X, Solids_log]
        else:
            return X


class PolyFeaturesAdder(BaseEstimator, TransformerMixin):
    '''Defines a transformer to add the features squared'''

    def __init__(self, degree=1):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.degree > 1:
            pf = PolynomialFeatures(
                degree=self.degree,
                interaction_only=False,
                include_bias=False
            )
            poly_feats = pf.fit_transform(X)
            return np.concatenate((X, poly_feats), axis=1)
        else:
            return X


def preprocessing_pipeline(
        missing="median",
        scaling="standard",
        add_Solids_log=False,
        poly_degree=1):
    """
    This function's goal is to build a preprocessing pipeline with given preprocessing strategy.

    Parameters
    ----------
    missing : string
        Specify the strategy for dealing with the missing values (default is "mean")
        Possible values: "mean", "median", “most_frequent”, "remove_rows", "remove_cols", "regression", "stochastic", "knn"
    scaling : string
        Specify the strategy for dealing with the scaling (default is "standard")
        Possible values: "standard", "min_max"
    add_Solids_log: boolean (default is True)
        Add log of solid feature
        Possible values: True, False
    poly_degree: int (default is 1)
        Add polynoms of features with the indicated degree
        Possible values: 1 (nothing is added), 2, 3...

    Returns
    -------
    sklearn.Pipeline
        The preprocessing pipeline with given strategies
    """
    # Missing
    if missing in ["mean", "median", "most_frequent"]:
        missing_imputer = SimpleImputer(strategy=missing)
    elif missing in ["remove_rows", "remove_cols"]:
        missing_imputer = RemoveNull(0 if missing == "remove_cols" else 1)
    elif missing in ["regression", "stochastic"]:
        missing_imputer = IterativeImputer(
            sample_posterior=(missing == "stochastic"))
    elif missing == "knn":
        missing_imputer = KNNImputer()

    # Added attributes
    solid_logs_adder = SolidsLogAdder(add_Solids_log=add_Solids_log)
    poly_features_adder = PolyFeaturesAdder(degree=poly_degree)

    # Scaling
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "min_max":
        scaler = MinMaxScaler(feature_range=(-1, 1))

    pipeline = Pipeline([
        ('missing', missing_imputer),
        ('solid_logs_adder', solid_logs_adder),
        ('sq_features_adder', poly_features_adder),
        ('scaling', scaler)
    ])
    return ModifiedPreprocessingPipeline(missing, pipeline)
