import pandas as pd
from sklearn import model_selection

# TODO: create a separate csv for test
if __name__ == "__main__":
    # Read training data
    df = pd.read_csv("./input/drinking_water_potability.csv")

    # fetch labels
    y = df["Potability"].values

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    # we use StratifiedKFold to keep the same % of targets per fold
    # (because of skewed targets)
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # save the new csv with kfold column
    df.to_csv("./input/dri_wat_pot_folds.csv", index=False)
