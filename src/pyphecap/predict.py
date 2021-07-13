from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from pyphecap.feature_matrix import build_feature_matrix, generate_feature_matrix
from pyphecap.phecap_data import Data
from pyphecap.sklearn_utils import build_classifier, get_auc
from pyphecap.surrogate import Surrogates


def predict_probabilities(data: Data, surrogates: Surrogates, coefficients: list, selected_features: list[str],
                          method='lasso_bic'):
    matrix = generate_feature_matrix(data, surrogates, selected_features)
    clf = build_classifier(coefficients, method=method)
    return clf.predict(matrix)


def predict_phenotype(data: Data, surrogates: Surrogates, coefficients: list, selected_features: list[str],
                      valid_roc, method='lasso_bic'):
    preds = predict_probabilities(data, surrogates, coefficients, selected_features, method=method)
    fprs = valid_roc.T[0]
    idx = np.argmin(abs(fprs - 0.05))
    cutoff_fpr95 = valid_roc[idx][2]  # threshold
    case_status = np.where(preds >= cutoff_fpr95, 1, 0)
    phenotype_df = pd.DataFrame({
        data.patient_id: data.frame[data.patient_id],
        'prediction': preds,
        'case_status': case_status,
    })
    return phenotype_df


def validate_phenotyping_model(data: Data, surrogates: Surrogates, coefficients: list, selected_features: list[str],
                               method='lasso_bic'):
    x, y_true = build_feature_matrix(data, surrogates, selected_features, is_validation=True)
    # TODO: subject weights: subject_weight <- data$subject_weight[ii]
    clf = build_classifier(coefficients, method=method)
    y_pred = clf.predict(x)
    roc = roc_auc_score(y_true, y_pred)
    auc = get_auc(y_true, y_pred)
    return roc, auc
