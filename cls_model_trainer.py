# cls_model_trainer.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import shap
import plotly.graph_objects as go

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
                'random_state': 42,
                'enable_categorical': True
            }
            if params:
                default_params.update(params)
            
            X_train_processed = X_train.copy()
            for col in X_train.select_dtypes(include=['object']).columns:
                X_train_processed[col] = X_train_processed[col].astype('category')
            
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

    @staticmethod
    def explain_forecast(model: Any, data: pd.DataFrame, model_type: str = "xgboost", feature_names: Optional[list] = None) -> Optional[go.Figure]:
        """
        Explain forecasts using SHAP and return a Plotly figure.

        Args:
            model: The trained model (XGBoost or DeepAR).
            data: The input data used for explanations (features).
            model_type: Type of the model ("xgboost" or "deepar").
            feature_names: Optional list of feature names for DeepAR.

        Returns:
            A Plotly Figure object for the SHAP summary plot or None if an error occurs.
        """
        try:
            if model_type == "xgboost":
                if not hasattr(model, 'predict'):
                    logger.error("Invalid XGBoost model for SHAP explanation.")
                    st.error("Invalid XGBoost model provided for SHAP explanation.", icon="🚨")
                    return None
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(data)
                # Compute mean absolute SHAP value for each feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_names = data.columns.tolist()
                fig = go.Figure(go.Bar(
                    x=mean_abs_shap,
                    y=feature_names,
                    orientation='h',
                    marker=dict(color=mean_abs_shap, colorscale='Blues'),
                ))
                fig.update_layout(
                    title="SHAP Feature Importance (XGBoost)",
                    xaxis_title="Mean |SHAP value|",
                    yaxis_title="Feature",
                    yaxis=dict(autorange="reversed"),
                    template="plotly_white"
                )
                logger.info(f"SHAP explanation generated for XGBoost model (Plotly).")
                return fig
            elif model_type == "deepar":
                if not hasattr(model, 'predict_for_shap'):
                    logger.error("DeepAR model does not have a 'predict_for_shap' method for SHAP KernelExplainer.")
                    st.error("DeepAR model is not configured for SHAP KernelExplainer. It needs a 'predict_for_shap' method.", icon="🚨")
                    return None
                if len(data) > 50:
                    data_sample = shap.sample(data, 50)
                else:
                    data_sample = data
                explainer = shap.KernelExplainer(model.predict_for_shap, data_sample)
                if feature_names:
                    data_for_shap = data[feature_names]
                else:
                    data_for_shap = data
                shap_values = explainer.shap_values(data_for_shap, nsamples=100)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_names = data_for_shap.columns.tolist()
                fig = go.Figure(go.Bar(
                    x=mean_abs_shap,
                    y=feature_names,
                    orientation='h',
                    marker=dict(color=mean_abs_shap, colorscale='Blues'),
                ))
                fig.update_layout(
                    title="SHAP Feature Importance (DeepAR)",
                    xaxis_title="Mean |SHAP value|",
                    yaxis_title="Feature",
                    yaxis=dict(autorange="reversed"),
                    template="plotly_white"
                )
                logger.info(f"SHAP explanation generated for DeepAR model (Plotly).")
                return fig
            else:
                logger.warning(f"Unsupported model type for SHAP: {model_type}")
                st.warning(f"Unsupported model type for SHAP: {model_type}", icon="⚠️")
                return None
        except Exception as e:
            logger.error(f"SHAP explanation failed for {model_type}: {str(e)}")
            st.error(f"SHAP explanation failed: {str(e)}", icon="🚨")
            return None