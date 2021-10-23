from sklearn import tree, ensemble

models = {
    "dt_gini": tree.DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    ),
    "dt_entropy": tree.DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
    ),
    "rf": ensemble.RandomForestClassifier(n_jobs=-1,
                                          random_state=42),
}
