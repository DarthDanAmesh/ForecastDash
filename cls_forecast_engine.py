import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
import logging
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastEngine:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def create_features(_df: pd.DataFrame, target_col: str, lags: int = 3, 
                       optional_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create time series features for supervised learning.
        
        Args:
            _df (pd.DataFrame): Input DataFrame with datetime index.
            target_col (str): Name of the target column.
            lags (int): Number of lag features to create. Defaults to 3.
            optional_features (List[str], optional): Additional feature columns to include.
        
        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        try:
            if _df.empty or target_col not in _df.columns:
                logger.error("Invalid input: DataFrame is empty or target column missing.")
                return pd.DataFrame()

            df = _df.copy()
            df['date'] = df.index
            df['dayofweek'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['dayofyear'] = df['date'].dt.dayofyear

            # Create lag features
            for lag in range(1, lags + 1):
                df[f'lag_{lag}'] = df[target_col].shift(lag)

            # Include optional features (e.g., promotions, holidays)
            if optional_features:
                for feature in optional_features:
                    if feature in df.columns:
                        df[feature] = df[feature]
                    else:
                        logger.warning(f"Optional feature '{feature}' not found in data.")

            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def train_arima(ts: pd.Series, order: tuple = (1, 1, 1)) -> Dict:
        """
        Train an ARIMA model on the time series data.
        
        Args:
            ts (pd.Series): Time series data with datetime index.
            order (tuple): ARIMA model order (p, d, q). Defaults to (1, 1, 1).
        
        Returns:
            Dict: Dictionary containing the trained model and metadata.
        """
        try:
            if ts is None or ts.empty:
                logger.error("Invalid time series for ARIMA training.")
                return {}

            model = ARIMA(ts, order=order)
            fitted_model = model.fit()
            return {
                'model': fitted_model,
                'last_values': ts.iloc[-3:].values,  # For compatibility with XGBoost
                'trained_at': pd.Timestamp.now()
            }
        except Exception as e:
            logger.error(f"ARIMA training failed: {str(e)}")
            return {}

    @staticmethod
    @st.cache_data(show_spinner=False)
    def forecast_xgboost(_model: object, _last_known_values: np.ndarray, periods: int, 
                        last_date: pd.Timestamp, freq: str = 'ME') -> Optional[pd.Series]:
        """
        Generate forecasts using an XGBoost model.
        
        Args:
            _model: Trained XGBoost model.
            _last_known_values (np.ndarray): Last known feature values.
            periods (int): Number of periods to forecast.
            last_date (pd.Timestamp): Last date in the historical data.
            freq (str): Frequency of the forecast dates. Defaults to 'ME'.
        
        Returns:
            pd.Series: Forecasted values with datetime index, or None if failed.
        """
        try:
            if periods <= 0 or _last_known_values.size == 0:
                logger.error("Invalid input: periods must be positive and last_known_values must not be empty.")
                return None

            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq=freq
            )
            forecast = []
            current_features = _last_known_values.copy()

            for _ in range(periods):
                next_pred = _model.predict(np.array(current_features).reshape(1, -1))[0]
                forecast.append(next_pred)
                current_features = np.roll(current_features, -1)
                current_features[-1] = next_pred

            return pd.Series(forecast, index=future_dates)
        except Exception as e:
            logger.error(f"XGBoost forecasting failed: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(show_spinner=False)
    def forecast_arima(_model: object, periods: int, last_date: Optional[pd.Timestamp] = None, 
                      freq: str = 'ME') -> Optional[pd.Series]:
        """
        Generate forecasts using an ARIMA model.
        
        Args:
            _model: Trained ARIMA model.
            periods (int): Number of periods to forecast.
            last_date (pd.Timestamp, optional): Last date in the historical data for indexing.
            freq (str): Frequency of the forecast dates. Defaults to 'ME'.
        
        Returns:
            pd.Series: Forecasted values with datetime index, or None if failed.
        """
        try:
            if periods <= 0:
                logger.error("Invalid input: periods must be positive.")
                return None

            forecast = _model.forecast(steps=periods)
            if last_date is not None:
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=periods,
                    freq=freq
                )
                forecast.index = future_dates
            return pd.Series(forecast)
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {str(e)}")
            return None