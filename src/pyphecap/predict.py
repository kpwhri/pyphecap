from sklearn.metrics import roc_auc_score

from pyphecap.feature_matrix import build_feature_matrix, generate_feature_matrix
from pyphecap.phecap_data import Data
from pyphecap.sklearn_utils import unpack_columns, build_classifier, get_auc
from pyphecap.surrogate import Surrogates


def predict_phenotype(data: Data, surrogates: Surrogates, coefficients: list, selected_features: list[str],
                      method='lasso_bic'):
    matrix = generate_feature_matrix(data, surrogates, selected_features)
    matrix = matrix[unpack_columns(coefficients)]
    clf = build_classifier(coefficients, method=method)
    preds = clf.predict(matrix)
    return preds


def validate_phenotyping_model(data: Data, surrogates: Surrogates, coefficients: list, selected_features: list[str],
                               method='lasso_bic'):
    x, y_true = build_feature_matrix(data, surrogates, selected_features, is_validation=False)
    x = x[unpack_columns(coefficients)]
    # TODO: subject weights: subject_weight <- data$subject_weight[ii]
    clf = build_classifier(coefficients, method=method)
    y_pred = clf.predict(x)
    roc = roc_auc_score(y_true, y_pred)
    auc = get_auc(y_true, y_pred)
    return roc, auc
