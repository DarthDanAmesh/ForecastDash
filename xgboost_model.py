# xgboost_model.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import logging
from constants import STANDARD_COLUMNS, DEFAULT_PREDICTION_LENGTH
from cls_model_trainer import ModelTrainer
from forecast_engine import ForecastEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, df: pd.DataFrame, forecast_horizon: int = DEFAULT_PREDICTION_LENGTH):
        """Initialize XGBoost model with dataset and parameters."""
        self.df = df
        self.model = None
        self.forecast_horizon = forecast_horizon
        self.feature_cols = ['month', 'quarter', 'day_of_week', 'is_weekend', 'delivery_delay'] + \
                           [f'lag_{i}' for i in range(1, 4)]
        logger.info("Initialized XGBoostModel")

    def train(self) -> bool:
        """Train the XGBoost model."""
        try:
            features_df = ForecastEngine.create_features(self.df)
            if features_df.empty:
                logger.error("Feature creation failed for XGBoost")
                st.error("Feature creation failed for XGBoost.", icon="🚨")
                return False

            X = features_df[self.feature_cols]
            y = features_df[STANDARD_COLUMNS['demand']]
            self.model = ModelTrainer.train_xgboost(X, y)
            if self.model:
                logger.info("XGBoost training completed")
                return True
            logger.error("XGBoost model training returned None")
            st.error("XGBoost model training failed.", icon="🚨")
            return False
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}", exc_info=True)
            st.error(f"XGBoost training failed: {str(e)}.", icon="🚨")
            self.model = None
            return False

    def predict(self, periods: int) -> Optional[pd.DataFrame]:
        """Generate forecasts using XGBoost."""
        try:
            if not self.model:
                logger.error("No trained XGBoost model available")
                st.error("No trained XGBoost model available.", icon="🚨")
                return None

            df = self.df.copy()
            max_date = df[STANDARD_COLUMNS['date']].max()
            future_dates = pd.date_range(
                start=max_date + pd.Timedelta(weeks=1),
                periods=periods,
                freq='W'
            )

            forecasts = []
            materials = df[STANDARD_COLUMNS['material']].unique()
            for material in materials:
                material_df = df[df[STANDARD_COLUMNS['material']] == material].copy()
                if material_df.empty:
                    logger.warning(f"No data for material {material}")
                    continue
                features_df = ForecastEngine.create_features(material_df)
                if features_df.empty:
                    logger.warning(f"Feature creation failed for material {material}")
                    continue
                last_features = features_df.tail(1)[self.feature_cols]
                
                material_forecasts = []
                current_features = last_features.values.copy()
                for i in range(periods):
                    pred = self.model.predict(current_features)[0]
                    material_forecasts.append(pred)
                    # Shift lags
                    current_features[:, -1] = current_features[:, -2]  # lag_3 = lag_2
                    current_features[:, -2] = current_features[:, -1]  # lag_2 = lag_1
                    current_features[:, -3] = pred  # lag_1 = prediction
                    # Update time-based features
                    current_features[:, 0] = (future_dates[i].month % 12) + 1  # month
                    current_features[:, 1] = (future_dates[i].month - 1) // 3 + 1  # quarter
                    current_features[:, 2] = future_dates[i].dayofweek  # day_of_week
                    current_features[:, 3] = 1 if future_dates[i].dayofweek >= 5 else 0  # is_weekend
                    # Keep delivery_delay constant (mean of historical)
                    current_features[:, 4] = features_df['delivery_delay'].mean()

                material_forecast_df = pd.DataFrame({
                    'date': future_dates,
                    STANDARD_COLUMNS['material']: material,
                    'forecast': material_forecasts,
                    'lower_bound': np.array(material_forecasts) * 0.9,
                    'upper_bound': np.array(material_forecasts) * 1.1
                })
                forecasts.append(material_forecast_df)

            if forecasts:
                forecast_df = pd.concat(forecasts, ignore_index=True)
                logger.info(f"XGBoost predictions generated successfully with {len(forecast_df)} rows")
                return forecast_df
            logger.warning("No XGBoost forecasts generated")
            return None
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {str(e)}", exc_info=True)
            st.error(f"XGBoost prediction failed: {str(e)}.", icon="🚨")
            return None