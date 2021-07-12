from sklearn.linear_model import LinearRegression

from pyphecap.phecap_data import Data
from pyphecap.surrogate import Surrogates
import pandas as pd


def build_feature_matrix(data: Data, surrogates: Surrogates, selected_features: list, is_validation=False):
    matrix = generate_feature_matrix(data, surrogates, selected_features)
    matrix[data.validation] = data.frame[data.validation]
    matrix[data.label] = data.frame[data.label]
    matrix = matrix[~pd.isnull(matrix[data.label])]
    is_validation = 1 if is_validation else 0
    y = matrix[matrix[data.validation] == is_validation][data.label]
    x = matrix[matrix[data.validation] == is_validation].drop(columns=[data.label, data.validation])
    return x, y


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
