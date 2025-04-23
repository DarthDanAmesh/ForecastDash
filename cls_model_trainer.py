import xgboost as xgb
import numpy as np
import streamlit as st
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_cache_load_util_funct import get_data_fingerprint, load_model_cache, save_model_cache

# Model Training Module
class ModelTrainer:
    @staticmethod
    def train_xgboost(X_train, y_train, params: Optional[Dict] = None, 
                     use_cache: bool = True, config: Optional[Dict] = None):
        """Train XGBoost with caching support"""
        if use_cache and config:
            fingerprint = get_data_fingerprint(pd.concat([X_train, y_train], axis=1), config)
            cached_model = load_model_cache(fingerprint, "xgboost")
            if cached_model:
                st.toast("Loaded cached XGBoost model")
                return cached_model
            
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

        if use_cache and config:
            save_model_cache(model, fingerprint, "xgboost")

        return model

    @staticmethod
    def train_arima(ts, order=(1, 1, 1), use_cache=True, config=None):
        """Train ARIMA model on time series data"""
        if use_cache and config:
            fingerprint = get_data_fingerprint(ts.to_frame(), config)
            cached_model = load_model_cache(fingerprint, "arima")
            if cached_model:
                st.toast("Loaded cached ARIMA model")
                return cached_model
            
        model = ARIMA(ts, order=order)
        model_fit = model.fit()
        if use_cache and config:
            save_model_cache(model_fit, fingerprint, "arima")
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