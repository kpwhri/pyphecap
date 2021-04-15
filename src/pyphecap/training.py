import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from pyphecap.phecap_data import Data
from pyphecap.surrogate import Surrogates


def train_phenotyping_model(data: Data, surrogates: Surrogates, selected_features,
                            method='lasso_bic', train_percent=0.7, num_splits=200):
    matrix = generate_feature_matrix(data, surrogates, selected_features)
    matrix[data.validation] = data.frame[data.validation]
    matrix[data.label] = data.frame[data.label]
    matrix = matrix[~pd.isnull(matrix[data.label])]
    y = matrix[matrix[data.validation] == 0][data.label]
    x = matrix[matrix[data.validation] == 0].drop(columns=[data.label, data.validation])
    coefficients, train_roc, train_auc, split_roc, split_auc = get_roc_auc(
        x, y, method=method, train_percent=train_percent

    )
    return coefficients, train_roc, split_roc


def get_roc_auc(x, y, method, train_percent):
    clf = LinearRegression()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    train_roc = roc_auc_score(y, y_pred)
    train_auc = roc_curve(y, y_pred)

    auc_data = []
    roc_data = []
    clf = LinearRegression()
    for i, (train_idx, test_idx) in enumerate(StratifiedShuffleSplit(n_splits=20, train_size=0.7).split(x, y)):
        clf.fit(x.iloc[train_idx], y.iloc[train_idx])
        y_pred_tr = clf.predict(x.iloc[train_idx])
        y_pred_te = clf.predict(x.iloc[test_idx])
        roc_curve(y.iloc[train_idx], y_pred_tr)
        curr_auc = []
        for fpr, tpr, thresh in zip(*roc_curve(y.iloc[test_idx], y_pred_te, drop_intermediate=False)):
            curr_auc.append((fpr, tpr, thresh))
        auc_data.append(np.array(curr_auc))
        roc_auc_score(y.iloc[train_idx], y_pred_tr)
        roc_auc = roc_auc_score(y.iloc[test_idx], y_pred_te)
        roc_data.append(roc_auc)

    coefficients = list(zip(['intercept'] + list(x.columns), [clf.intercept_] + clf.coef_))
    split_auc = sum(auc_data) / len(auc_data)
    split_roc = sum(roc_data)
    return coefficients, train_roc, train_auc, split_roc, split_auc


def generate_feature_matrix(data: Data, surrogates: Surrogates, selected_features):
    surrogate_matrix = pd.DataFrame({
        '__'.join(surrogate.variable_names): data.feature_transformation(
            data.frame[surrogate.variable_names].sum(axis=1)
        ) for surrogate in surrogates
    })
    surrogate_matrix[data.hu_feature] = data.feature_transformation(
        data.frame[data.hu_feature]
    )
    other_matrix = pd.DataFrame({
        col: data.feature_transformation(data.frame[col])
        for col in set(selected_features) - set(surrogate_matrix.columns)
    })
    lr = LinearRegression()
    lr.fit(surrogate_matrix, other_matrix)
    pred = lr.predict(surrogate_matrix)
    residual = (other_matrix - pred)  # get residual
    for col in residual.columns:
        surrogate_matrix[col] = residual[col]
    return surrogate_matrix
