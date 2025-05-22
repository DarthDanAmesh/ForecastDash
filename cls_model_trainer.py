# cls_model_trainer.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    @staticmethod
    def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, 
                     params: Optional[Dict] = None) -> Any:
        """Train XGBoost model with default parameters."""
        try:
            from xgboost import XGBRegressor
            default_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
            if params:
                default_params.update(params)
            
            model = XGBRegressor(**default_params)
            model.fit(X_train, y_train)
            logger.info("XGBoost model trained successfully")
            return model
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            st.error(f"XGBoost training failed: {str(e)}.", icon="🚨")
            return None

    @staticmethod
    def train_deepar(df: pd.DataFrame, params: Optional[Dict] = None) -> Any:
        """Train DeepAR model using pytorch_forecasting."""
        try:
            from deepar_model import DeepARModel
            model = DeepARModel(df, params)
            success = model.train()
            if success:
                logger.info("DeepAR model trained successfully")
                return model
            logger.error("DeepAR training did not complete successfully")
            st.error("DeepAR training did not complete successfully.", icon="🚨")
            return None
        except Exception as e:
            logger.error(f"DeepAR training failed: {str(e)}")
            st.error(f"DeepAR training failed: {str(e)}.", icon="🚨")
            return None

    @staticmethod
    def evaluate_model(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance using MAE, MSE, and RMSE."""
        try:
            mae = mean_absolute_error(actuals, predictions)
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            return {'mae': mae, 'mse': mse, 'rmse': rmse}
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            st.error(f"Model evaluation failed: {str(e)}.", icon="🚨")
            return {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan}