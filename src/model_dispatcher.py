from sklearn import tree, ensemble

models = {
    "dt_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "dt_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier(),
}
