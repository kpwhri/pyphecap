from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

from pyphecap.feature_matrix import build_feature_matrix
from pyphecap.phecap_data import Data
from pyphecap.sklearn_utils import get_auc, pack_intercept_and_coefficients
from pyphecap.surrogate import Surrogates


def train_phenotyping_model(data: Data, surrogates: Surrogates, selected_features,
                            method='lasso_bic', train_percent=0.7, num_splits=200):
    x, y = build_feature_matrix(data, surrogates, selected_features, is_validation=False)
    # TODO: subject weights: subject_weight <- data$subject_weight[ii]
    # TODO: penalty weights: penalty_weight <- c(
    #     rep.int(0.0, attr(feature, "free")),
    #     rep.int(1.0, ncol(feature) - attr(feature, "free")))
    coefficients, train_roc, train_auc, split_roc, split_auc = get_roc_auc(
        x, y, method=method, train_percent=train_percent
    )
    return coefficients, (train_roc, train_auc), (split_roc, split_auc)


def get_roc_auc(x, y, method, train_percent):
    clf = LinearRegression()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    train_roc = roc_auc_score(y, y_pred)
    train_auc = get_auc(y, y_pred)

    auc_data = []
    roc_data = []
    clf = LinearRegression()
    for i, (train_idx, test_idx) in enumerate(StratifiedShuffleSplit(n_splits=20, train_size=0.7).split(x, y)):
        clf.fit(x.iloc[train_idx], y.iloc[train_idx])
        y_pred_tr = clf.predict(x.iloc[train_idx])
        y_pred_te = clf.predict(x.iloc[test_idx])
        roc_curve(y.iloc[train_idx], y_pred_tr)
        auc_data.append(get_auc(y.iloc[test_idx], y_pred_te))
        roc_auc_score(y.iloc[train_idx], y_pred_tr)
        roc_auc = roc_auc_score(y.iloc[test_idx], y_pred_te)
        roc_data.append(roc_auc)

    coefficients = pack_intercept_and_coefficients(clf, x)
    split_auc = sum(auc_data) / len(auc_data)
    split_roc = sum(roc_data) / len(roc_data)
    return coefficients, train_roc, train_auc, split_roc, split_auc
