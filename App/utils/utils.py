from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CreateAdditionalAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, get_bedrooms_per_room=True):
        self.get_bedrooms_per_room = get_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        population_per_household = X.iloc[:, 5] / X.iloc[:, 6]
        rooms_per_household = X.iloc[:, 3] / X.iloc[:, 6]
        if self.get_bedrooms_per_room:
            bedrooms_per_room = X.iloc[:, 4] / X.iloc[:, 3]
            return np.c_[X, population_per_household, rooms_per_household, bedrooms_per_room]
        else:
            return np.c_[X, population_per_household, rooms_per_household]