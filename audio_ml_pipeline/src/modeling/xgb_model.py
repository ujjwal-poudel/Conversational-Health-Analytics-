import xgboost as xgb
import numpy as np
from typing import Dict, Any


class XGBoostModel:
    """
    Wrapper around XGBoost's regressor to standardize training and prediction.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the model with given parameters.

        Args:
            params: Dictionary of XGBoost hyperparameters.
        """
        self.params = params
        self.model = None

    def train(self, X_train, y_train):
        """
        Train the XGBoost model on provided training data.
        """
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict PHQ scores for the given feature matrix.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    def get_feature_importance(self, feature_names):
        """
        Return a dictionary mapping feature names to importance values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))