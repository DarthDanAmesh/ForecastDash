import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Model Training Module
class ModelTrainer:
    @staticmethod
    def train_xgboost(X_train, y_train, params: Optional[Dict] = None):
        """Train XGBoost model with default or custom parameters"""
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
            
        model = xgb.XGBRegressor(**default_params)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def train_arima(ts, order=(1, 1, 1)):
        """Train ARIMA model on time series data"""
        model = ARIMA(ts, order=order)
        model_fit = model.fit()
        return model_fit

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'predictions': predictions
        }