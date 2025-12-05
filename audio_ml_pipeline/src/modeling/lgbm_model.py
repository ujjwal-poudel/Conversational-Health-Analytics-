import lightgbm as lgb
import numpy as np
from typing import Dict, Any


class LightGBMModel:
    """
    Wrapper around LightGBM's regressor to standardize training and prediction.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the model with given parameters.
        
        Args:
            params: Dictionary of LightGBM hyperparameters.
        """
        self.params = params
        self.model = None

    def train(self, X_train, y_train):
        """
        Train the LightGBM model on provided training data.
        """
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict PHQ scores for given feature matrix.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    def get_feature_importance(self, feature_names):
        """
        Return a sorted dictionary of feature importance scores.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))