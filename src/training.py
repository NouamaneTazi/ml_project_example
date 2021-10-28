from sklearn import metrics


def train(x_train, y_train, x_valid, y_valid, model):
    """
    Fits :model: on :x_train:, and validates on :x_valid:.
    :param x_train: the numpy array with train data
    :param y_train: the numpy array with train labels
    :param x_valid: the numpy array with valid data
    :param y_valid: the numpy array with valid labels
    :param fold: the fold id used for validation
    :param model: the model (already built)
    :return: a dict containing the trained model and metrics (accuracy, AUC, f1 score)
    """
    # fit the model on training data
    model.fit(x_train, y_train)
    # create predictions for validation samples
    valid_preds = model.predict(x_valid)
    valid_probs = model.predict_proba(x_valid)
    # get roc auc and accuracy and f1 score
    accuracy = metrics.accuracy_score(y_valid, valid_preds)
    auc = metrics.roc_auc_score(
        y_valid, valid_probs[:, 1])
    f1_score = metrics.f1_score(y_valid, valid_preds)
    return {"model": model, "metrics": {"accuracy": accuracy, "auc": auc, "f1_score": f1_score}}