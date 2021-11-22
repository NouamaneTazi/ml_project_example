""" create folds for cross-validation """

from src import config
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

if __name__ == "__main__":
    # Read training data
    df = pd.read_csv(f"./{config.RAW_FILE}")

    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # split train/test
    df_train, df_test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["Potability"], random_state=config.RANDOM_STATE)
    print(df_train.head())

    # we create a new column called kfold and fill it with -1
    df_train.loc[:, "kfold"] = -1

    # initiate the kfold class from model_selection module
    # we use StratifiedKFold to keep the same % of targets per fold
    # (because of skewed targets)
    kf = StratifiedKFold(n_splits=config.NUMBER_FOLDS)

    # shuffle data
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(
            kf.split(X=df_train, y=df_train["Potability"])):
        df_train.loc[v_, 'kfold'] = f

    # save the new csvs with kfold column
    df_train.to_csv(f"./{config.TRAINING_FILE}", index=False)
    df_test.to_csv(f"./{config.TESTING_FILE}", index=False)
