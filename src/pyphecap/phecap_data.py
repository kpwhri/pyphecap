import random
from dataclasses import dataclass, field, InitVar

import numpy as np
import pandas as pd


def add_validation_column(df: pd.DataFrame, percent_validation=0.4, column_label='validation'):
    df[column_label] = [1 if random.random() < percent_validation else 0 for _ in range(df.shape[0])]


@dataclass(frozen=True)
class Data:
    frame: pd.DataFrame  # TODO: PheCAP allows this to be a list, or a path
    hu_feature: str  # utilization column name
    label: str  # gold standard label column name
    validation: str  # column name with 1=valid, 0=train
    patient_id: str = 'patient_id'  # patient id column
    subject_weight: list[float] = None  # list of weights for each subject
    feature_transformation = np.log1p  # or is it a function?
    training_set: pd.DataFrame = None
    validation_set: pd.DataFrame = None

    def __post_init__(self):
        if not self.subject_weight:
            object.__setattr__(self, 'subject_weight', [1.0] * self.frame.shape[1])
        if self.hu_feature not in self.frame.columns:
            raise ValueError(f'`hu_feature` column name "{self.hu_feature}" not in frame.columns')
        if self.label not in self.frame.columns:
            raise ValueError(f'`label` column name "{self.label}" not in frame.columns')
        # split valid/train
        object.__setattr__(self, 'training_set', self.frame[self.frame[self.validation] == 0].copy())
        object.__setattr__(self, 'validation_set', self.frame[self.frame[self.validation] == 1].copy())

    @property
    def frame_no_label(self):
        return self.frame.drop(columns=[self.label, self.validation])
