import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.impute import SimpleImputer


def run(fold):
    # read the training data with folds
    df = pd.read_csv("input/dri_wat_pot_folds.csv")

    # handle missing data
    # fill all missing data with -1 and let decision tree handle it
    imp_constant = SimpleImputer(strategy='constant', fill_value=-1)
    df = pd.DataFrame(imp_constant.fit_transform(df), columns=df.columns)

    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the Potability column from dataframe and convert it to
    # a numpy array by using .values.
    # target is Potability column in the dataframe
    x_train = df_train.drop("Potability", axis=1).values
    y_train = df_train.Potability.values
    # similarly, for validation, we have
    x_valid = df_valid.drop("Potability", axis=1).values
    y_valid = df_valid.Potability.values

    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()
    # fit the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    valid_preds = clf.predict(x_valid)

    # get roc auc and accuracy score
    auc = metrics.roc_auc_score(df_valid.Potability.values, valid_preds)
    accuracy = metrics.accuracy_score(y_valid, valid_preds)
    print(f"Fold={fold}, Accuracy={accuracy}, AUC={auc}")
    # save the model
    joblib.dump(clf, f"models/dt_{fold}.bin")


if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
