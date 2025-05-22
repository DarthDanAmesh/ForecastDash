# forecast_engine.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from constants import STANDARD_COLUMNS, DEFAULT_PREDICTION_LENGTH, DEFAULT_FREQ

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastEngine:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def create_features(_df: pd.DataFrame, target_col: str = STANDARD_COLUMNS['demand'], 
                       lags: int = 3) -> pd.DataFrame:
        """Create features for XGBoost forecasting."""
        try:
            if _df.empty or target_col not in _df.columns:
                logger.error("Invalid input: DataFrame empty or target column missing")
                st.error("Invalid data for feature creation.", icon="🚨")
                return pd.DataFrame()

            df = _df.copy()
            for lag in range(1, lags + 1):
                df[f'lag_{lag}'] = df[target_col].shift(lag)
            df['month'] = df[STANDARD_COLUMNS['date']].dt.month
            df['quarter'] = df[STANDARD_COLUMNS['date']].dt.quarter
            df['day_of_week'] = df[STANDARD_COLUMNS['date']].dt.dayofweek
            df['is_weekend'] = df[STANDARD_COLUMNS['date']].dt.dayofweek.isin([5, 6]).astype(int)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Feature creation failed: {str(e)}")
            st.error(f"Feature creation failed: {str(e)}.", icon="🚨")
            return pd.DataFrame()

    @staticmethod
    def forecast(df: pd.DataFrame, forecast_horizon: int = DEFAULT_PREDICTION_LENGTH, 
                 freq: str = DEFAULT_FREQ) -> Optional[pd.DataFrame]:
        """Generate forecasts using DeepAR, falling back to XGBoost if DeepAR fails."""
        try:
            if df.empty or STANDARD_COLUMNS['material'] not in df.columns:
                logger.error("Invalid input: DataFrame empty or material column missing")
                st.error("Data must include material column for SKU-level forecasting.", icon="🚨")
                return None

            from deepar_model import DeepARModel
            from xgboost_model import XGBoostModel
            from cls_model_trainer import ModelTrainer

            # Try DeepAR
            deepar_params = {
                'max_epochs': 10,
                'batch_size': 32,
                'prediction_length': forecast_horizon
            }
            deepar_model = ModelTrainer.train_deepar(df, deepar_params)
            if deepar_model:
                forecasts = deepar_model.predict(df, periods=forecast_horizon)
                if forecasts is not None and not forecasts.empty:
                    logger.info("DeepAR forecast generated successfully")
                    return forecasts

            # Fallback to XGBoost
            st.warning("DeepAR failed, falling back to XGBoost.", icon="⚠️")
            xgb_model = XGBoostModel(df, forecast_horizon=forecast_horizon)
            xgb_model.train()
            forecasts = xgb_model.predict(periods=forecast_horizon)
            if forecasts is not None and not forecasts.empty:
                logger.info("XGBoost forecast generated successfully")
                return forecasts

            logger.error("Both DeepAR and XGBoost failed to generate forecasts")
            st.error("Failed to generate forecasts.", icon="🚨")
            return None
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            st.error(f"Forecasting failed: {str(e)}.", icon="🚨")
            return None