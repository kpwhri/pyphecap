import math
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pyphecap.phecap_data import Data
from pyphecap.surrogate import Surrogates


def run_feature_extraction(data: Data, surrogates: Surrogates, subsample_size: int = 1000, num_subsamples: int = 200,
                           dropout_proportion: float = 0, frequency_cutoff: float = 0.5):
    surrogate_cols = set()
    half_subsample_size = int(subsample_size / 2)
    for surrogate in surrogates:
        surrogate_cols |= set(surrogate.variable_names)
        series = data.frame[surrogate.variable_names].sum(axis=1)
        cases_mask = series > surrogate.upper_cutoff
        controls_mask = series < surrogate.lower_cutoff
        if cases_mask.sum() <= half_subsample_size:
            raise ValueError(
                f'Too few cases ({cases_mask.sum()}): decrease upper cutoff ({surrogate.upper_cutoff})'
                f' or sample size ({subsample_size})')
        if controls_mask.sum() <= half_subsample_size:
            raise ValueError(
                f'Too few controls ({controls_mask.sum()}): increase lower cutoff ({surrogate.lower_cutoff})'
                f' or sample size ({subsample_size})')

    df = data.frame_no_label.apply(data.feature_transformation).drop(columns=surrogate_cols)
    freqs = []
    for surrogate in surrogates:
        series = data.frame[surrogate.variable_names].sum(axis=1)
        cases_mask = series > surrogate.upper_cutoff
        controls_mask = series < surrogate.lower_cutoff

        cases = df[cases_mask]
        controls = df[controls_mask]

        lr = LinearRegression(positive=True, )

        y = [1.0] * half_subsample_size + [0.0] * half_subsample_size
        freqs.append(
            sum(run_model(lr, y, cases, controls, half_subsample_size, dropout_proportion)
                for _ in range(num_subsamples))
        )
    selected_features = np.where(sum(freqs) >= num_subsamples * len(surrogates) * frequency_cutoff, 1, 0)
    selected_labels = list(surrogate_cols) + [col for feat, col in zip(selected_features, df.columns) if feat]
    return selected_labels


def run_model(model, y, cases, controls, half_subsample_size, dropout_proportion):
    """
    TODO: implement dropout
    :param model:
    :param y:
    :param cases:
    :param controls:
    :param half_subsample_size:
    :param dropout_proportion:
    :return:
    """
    case_sample = cases.sample(n=half_subsample_size)
    control_sample = controls.sample(n=half_subsample_size)
    sample = pd.concat((case_sample, control_sample))
    if dropout_proportion:
        sample = dropout(sample, dropout_proportion)
    model.fit(sample, y)
    return np.where(model.coef_ > 0.0, 1, 0)  # does not include intercept


def dropout(df, dropout_proportion):
    """
    Set specified proportion to 0
    :param df:
    :param dropout_proportion: for each column, set this percent of values to 0
    :return:
    """
    count = math.ceil(dropout_proportion * df.shape[1])
    for col in df.columns:
        df.loc[random.sample(df.index, k=count), col] = 0
    return df
