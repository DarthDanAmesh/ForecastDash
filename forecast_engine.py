# forecast_engine.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from constants import STANDARD_COLUMNS, DEFAULT_PREDICTION_LENGTH, DEFAULT_FREQ
from cls_model_trainer import ModelTrainer


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
            
            # Ensure we have a unique index by including material if it exists
            if STANDARD_COLUMNS['material'] in df.columns:
                # Sort by date and material
                df = df.sort_values([STANDARD_COLUMNS['material'], STANDARD_COLUMNS['date']])
                
                # Process each material separately
                result_dfs = []
                for material, group in df.groupby(STANDARD_COLUMNS['material']):
                    group_df = group.copy()
                    
                    # Create a rank-based time index within each group
                    group_df['time_rank'] = group_df[STANDARD_COLUMNS['date']].rank(method='dense').astype(int)
                    
                    # Create lags using the rank-based approach instead of frequency-based shifting
                    for lag in range(1, lags + 1):
                        group_df[f'lag_{lag}'] = group_df[target_col].shift(lag)
                    
                    # Add date-based features
                    group_df['month'] = pd.to_datetime(group_df[STANDARD_COLUMNS['date']]).dt.month
                    group_df['quarter'] = pd.to_datetime(group_df[STANDARD_COLUMNS['date']]).dt.quarter
                    group_df['day_of_week'] = pd.to_datetime(group_df[STANDARD_COLUMNS['date']]).dt.dayofweek
                    group_df['is_weekend'] = group_df['day_of_week'].isin([5, 6]).astype(int)
                    
                    # Drop the temporary rank column
                    group_df = group_df.drop(columns=['time_rank'])
                    result_dfs.append(group_df)
                
                # Combine all processed groups
                if result_dfs:
                    df = pd.concat(result_dfs, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                # If no material column, just sort by date
                df = df.sort_values(STANDARD_COLUMNS['date'])
                
                # Create a rank-based time index
                df['time_rank'] = df[STANDARD_COLUMNS['date']].rank(method='dense').astype(int)
                
                # Create lags using the rank-based approach
                for lag in range(1, lags + 1):
                    df[f'lag_{lag}'] = df[target_col].shift(lag)
                
                # Add date-based features
                df['month'] = pd.to_datetime(df[STANDARD_COLUMNS['date']]).dt.month
                df['quarter'] = pd.to_datetime(df[STANDARD_COLUMNS['date']]).dt.quarter
                df['day_of_week'] = pd.to_datetime(df[STANDARD_COLUMNS['date']]).dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                
                # Drop the temporary rank column
                df = df.drop(columns=['time_rank'])
            
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Feature creation failed: {str(e)}")
            st.error(f"Feature creation failed: {str(e)}.", icon="🚨")
            return pd.DataFrame()

    @staticmethod
    def forecast(df: pd.DataFrame, forecast_horizon: int = DEFAULT_PREDICTION_LENGTH, 
                 freq: str = DEFAULT_FREQ) -> Optional[dict]:
        """Generate forecasts using DeepAR, falling back to XGBoost if DeepAR fails or encounters unknown categories."""
        try:
            if df.empty or STANDARD_COLUMNS['material'] not in df.columns:
                logger.error("Invalid input: DataFrame empty or material column missing")
                st.error("Data must include material column for SKU-level forecasting.", icon="🚨")
                return None

            # Try DeepAR first
            deepar_params = {
                'max_epochs': 10,
                'batch_size': 32,
                'prediction_length': forecast_horizon
            }
            deepar_model = ModelTrainer.train_deepar(df, deepar_params)
            
            if deepar_model:
                # Split data by known/unknown categories if DeepAR model is available
                deepar_forecasts = deepar_model.predict(df, periods=forecast_horizon)
                
                if deepar_forecasts is not None and not deepar_forecasts.empty:
                    logger.info("DeepAR forecast generated successfully")
                    feature_names = getattr(deepar_model, 'feature_names_', None)
                    features_for_shap = df.drop(columns=[STANDARD_COLUMNS['demand']]) if STANDARD_COLUMNS['demand'] in df.columns else df
                    return {
                        'forecast': deepar_forecasts,
                        'model': deepar_model,
                        'features': features_for_shap,
                        'model_type': 'deepar',
                        'feature_names': feature_names
                    }
                else:
                    logger.warning("DeepAR couldn't generate forecasts, possibly due to unknown categories")
                
                # Fallback to XGBoost
                st.warning("DeepAR failed or encountered unknown categories, using XGBoost instead.", icon="⚠️")
            # Create features for XGBoost
            df_feat = ForecastEngine.create_features(df, target_col=STANDARD_COLUMNS['demand'])
            if df_feat.empty:
                logger.error("Feature creation for XGBoost failed.")
                st.error("Feature creation for XGBoost failed.", icon="🚨")
                return None
            X = df_feat.drop(columns=[STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['date']])
            y = df_feat[STANDARD_COLUMNS['demand']]
            xgb_model = ModelTrainer.train_xgboost(X, y)
            if xgb_model:
                from xgboost import XGBRegressor
                # Predict using XGBoostModel wrapper if needed, else use model directly
                xgb_forecast = None
                try:
                    from deepar_model import DeepARModel
                    from xgboost_model import XGBoostModel
                    # Try to use the XGBoostModel wrapper if available
                    xgb_model_wrapper = XGBoostModel(df, forecast_horizon=forecast_horizon)
                    xgb_model_wrapper.model = xgb_model
                    xgb_forecast = xgb_model_wrapper.predict(periods=forecast_horizon)
                except Exception as e:
                    logger.warning(f"XGBoostModel wrapper failed: {str(e)}. Using model.predict instead.")
                    # Fallback: just use model.predict for the last available row
                    last_row = X.iloc[[-1]]
                    preds = xgb_model.predict(last_row)
                    xgb_forecast = pd.DataFrame({
                        STANDARD_COLUMNS['date']: [df[STANDARD_COLUMNS['date']].max() + pd.Timedelta(weeks=i+1) for i in range(forecast_horizon)],
                        'forecast': preds[0] if hasattr(preds, '__getitem__') else preds
                    })
                if xgb_forecast is not None and not xgb_forecast.empty:
                    logger.info("XGBoost forecast generated successfully")
                    return {
                        'forecast': xgb_forecast,
                        'model': xgb_model,
                        'features': X,
                        'model_type': 'xgboost',
                        'feature_names': X.columns.tolist()
                    }

            logger.error("Both DeepAR and XGBoost failed to generate forecasts")
            st.error("Failed to generate forecasts.", icon="🚨")
            return None
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            st.error(f"Forecasting failed: {str(e)}.", icon="🚨")
            return None