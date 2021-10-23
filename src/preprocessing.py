from sklearn import preprocessing
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


def add_polynomial(df):
    # initialize polynomial features class object
    # for two-degree polynomial features
    pf = preprocessing.PolynomialFeatures(
        degree=2,
        interaction_only=False,
        include_bias=False
    )
    # fit to the features
    pf.fit(df)
    # create polynomial features
    poly_feats = pf.transform(df)
    # create a dataframe with all the features
    num_feats = poly_feats.shape[1]
    df_transformed = pd.DataFrame(
        poly_feats,
        columns=[f"f_{i}" for i in range(1, num_feats + 1)]
    )
    return df


class RemoveNull(BaseEstimator, TransformerMixin):
    '''Defines a transformer to delete rows or cols containing null values'''

    def __init__(self, direction=0):
        self.direction = direction

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.dropna(axis=self.direction)


class AttributesAdder(BaseEstimator, TransformerMixin):
    '''Defines a transformer to add custom features'''

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


def preprocessing_pipeline(missing="median", scaling="standard", add_Solids_log=True):
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

    Returns
    -------
    sklearn.Pipeline
        The preprocessing pipeline with given strategies
    """
    # Missing
    if missing in ["mean", "median", "most_frequent"]:
        missing_imputer = SimpleImputer(strategy=missing)
    elif missing in ["remove_rows", "remove_cols"]:
        missing_imputer = RemoveNull(0 if missing == "remove_rows" else 1)
    elif missing in ["regression", "stochastic"]:
        missing_imputer = IterativeImputer(
            sample_posterior=(missing == "stochastic"))
    elif missing == "knn":
        missing_imputer = KNNImputer()

    # Added attributes
    attr_adder = AttributesAdder(add_Solids_log=add_Solids_log)

    # Scaling
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "min_max":
        scaler = MinMaxScaler(feature_range=(-1, 1))

    return Pipeline([
        ('missing', missing_imputer),
        ('attribs_adder', attr_adder),
        ('scaling', scaler)
    ])
