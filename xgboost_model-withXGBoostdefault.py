# xgboost_model.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import logging
from constants import STANDARD_COLUMNS, DEFAULT_FORECAST_PERIOD
from cls_model_trainer import ModelTrainer
from forecast_engine import ForecastEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, df: pd.DataFrame, forecast_horizon: int = DEFAULT_FORECAST_PERIOD):
        """Initialize XGBoost model with dataset and parameters."""
        self.df = df
        self.model = None
        self.forecast_horizon = forecast_horizon
        self.trained_features = None  # Store the actual features used during training
        logger.info("Initialized XGBoostModel")

    def _encode_categorical_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features to numeric codes consistently."""
        df = features_df.copy()
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding categorical columns: {list(categorical_cols)}")
            # Convert categorical columns to numeric using label encoding
            for col in categorical_cols:
                if col not in [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand']]:
                    if df[col].dtype == 'category':
                        df[col] = df[col].cat.codes
                    else:  # object type
                        df[col] = pd.Categorical(df[col]).codes
        
        # Ensure all columns are numeric (except date and target)
        exclude_cols = [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand']]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        keep_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        return df[keep_cols + [col for col in exclude_cols if col in df.columns]]

    def train(self) -> bool:
        """Train the XGBoost model."""
        try:
            features_df = ForecastEngine.create_features(self.df)
            if features_df.empty:
                logger.error("Feature creation failed for XGBoost")
                st.error("Feature creation failed for XGBoost.", icon="🚨")
                return False

            # Encode categorical features
            processed_df = self._encode_categorical_features(features_df)
            
            # Prepare features and target
            X = processed_df.drop(columns=[STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['date']], errors='ignore')
            y = features_df[STANDARD_COLUMNS['demand']]
            
            # Store the feature columns for consistent prediction
            self.trained_features = X.columns.tolist()
            
            self.model = ModelTrainer.train_xgboost(X, y)
            if self.model:
                logger.info(f"XGBoost training completed with features: {self.trained_features}")
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
            if not self.model or not self.trained_features:
                logger.error("No trained XGBoost model or feature info available")
                st.error("No trained XGBoost model available.", icon="🚨")
                return None

            df = self.df.copy()
            max_date = df[STANDARD_COLUMNS['date']].max()
            future_dates = pd.date_range(
                start=max_date + pd.offsets.MonthBegin(1),
                periods=periods,
                freq='ME'
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
                
                # Process features the same way as during training
                processed_df = self._encode_categorical_features(features_df)
                
                # Get the last row of features
                X_last = processed_df.drop(columns=[STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['date']], errors='ignore')
                
                # Ensure we have the same features as training
                missing_features = set(self.trained_features) - set(X_last.columns)
                if missing_features:
                    logger.warning(f"Missing features for material {material}: {missing_features}")
                    # Add missing features with default values
                    for feat in missing_features:
                        X_last[feat] = 0
                
                # Select only the trained features in the same order
                X_last = X_last[self.trained_features]
                last_features = X_last.tail(1).copy()
                
                material_forecasts = []
                current_features = last_features.copy()
                
                for i in range(periods):
                    # Make prediction
                    pred = self.model.predict(current_features)[0]
                    material_forecasts.append(pred)
                    
                    # Update lag features for next prediction if they exist
                    if 'lag_3' in current_features.columns and 'lag_2' in current_features.columns:
                        current_features['lag_3'] = current_features['lag_2'].iloc[0]
                    if 'lag_2' in current_features.columns and 'lag_1' in current_features.columns:
                        current_features['lag_2'] = current_features['lag_1'].iloc[0]
                    if 'lag_1' in current_features.columns:
                        current_features['lag_1'] = pred
                    
                    # Update date-based features
                    if 'month' in current_features.columns:
                        current_features['month'] = future_dates[i].month
                    if 'quarter' in current_features.columns:
                        current_features['quarter'] = (future_dates[i].month - 1) // 3 + 1
                    if 'day_of_week' in current_features.columns:
                        current_features['day_of_week'] = future_dates[i].dayofweek
                    if 'is_weekend' in current_features.columns:
                        current_features['is_weekend'] = 1 if future_dates[i].dayofweek >= 5 else 0

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