
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Clip every column to [Q1 - k*IQR, Q3 + k*IQR] learned from the training data."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.k * iqr).values
        self.upper_ = (q3 + self.k * iqr).values
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return pd.DataFrame(np.clip(X.values, self.lower_, self.upper_),
                            columns=self.columns_, index=X.index)
