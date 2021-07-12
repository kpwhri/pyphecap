import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve


def pack_intercept_and_coefficients(clf, x):
    return list(zip(['intercept'] + list(x.columns), [clf.intercept_] + clf.coef_))


def unpack_coefficients(intercept_and_coefficients):
    return np.array([x[1] for x in intercept_and_coefficients[1:]])


def unpack_intercept(intercept_and_coefficients):
    return intercept_and_coefficients[0][1]


def unpack_columns(intercept_and_coefficients):
    return [x[0] for x in intercept_and_coefficients[1:]]


def build_classifier(coefficients, method='lasso_bic', **kwargs):
    clf = get_model(method, **kwargs)
    clf.intercept_ = unpack_intercept(coefficients)
    clf.coef_ = unpack_coefficients(coefficients)
    return clf


def get_model(method, **kwargs):
    return {
        'lasso_bic': LinearRegression,
    }.get(method, LinearRegression)(**kwargs)


def get_auc(y_true, y_pred):
    curr_auc = []
    for fpr, tpr, thresh in zip(*roc_curve(y_true, y_pred, drop_intermediate=False)):
        curr_auc.append((fpr, tpr, thresh))
    return np.array(curr_auc)
