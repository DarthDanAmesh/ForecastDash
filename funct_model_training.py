import streamlit as st
from typing import Optional, Dict, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from consts_model import MODEL_TYPES, DEFAULT_FORECAST_PERIOD
from cls_forecast_engine import ForecastEngine
from cls_data_preprocessor import DataProcessor
from cls_model_trainer import ModelTrainer
from funct_plot_predictions import plot_predictions
from cls_session_management import SessionState
from column_config import STANDARD_COLUMNS

def validate_training_data(ts: pd.Series, min_length: int = 12) -> Tuple[bool, str]:
    """
    Validate the time series data for model training.

    Args:
        ts (pd.Series): Time series data.
        min_length (int): Minimum number of data points required.

    Returns:
        Tuple[bool, str]: (is_valid, message) indicating if the data is valid and any error message.
    """
    if ts is None:
        return False, (
            "Time series data is None. Ensure the dataset includes 'date' and 'demand' columns. "
            "Supported aliases include 'quantity' for 'demand'."
        )
    if not isinstance(ts, pd.Series):
        return False, "Invalid data: Must be a Pandas Series."
    if len(ts) < min_length:
        return False, f"Insufficient data: At least {min_length} data points required, got {len(ts)}."
    if not pd.api.types.is_numeric_dtype(ts):
        return False, "Time series must contain numeric values."
    if ts.isna().any():
        return False, "Time series contains missing values. Please preprocess the data."
    return True, ""

@st.cache_data(show_spinner=False)
def prepare_training_data(_df: pd.DataFrame, target_col: str = STANDARD_COLUMNS['demand'], 
                         date_col: str = STANDARD_COLUMNS['date'], freq: str = 'ME') -> Optional[pd.Series]:
    """
    Prepare time series data for model training.

    Args:
        _df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column for time series (default: 'demand').
        date_col (str): Date column for time series (default: 'date').
        freq (str): Resampling frequency (default: 'ME' for month-end).

    Returns:
        Optional[pd.Series]: Prepared time series or None if preparation fails.
    """
    try:
        if _df is None or _df.empty:
            st.error(
                "No data provided. Please upload a dataset with 'date' and 'demand' columns.", 
                icon="🚨"
            )
            return None

        with st.spinner("Preparing time series data..."):
            ts = DataProcessor.prepare_time_series(_df, target_col, date_col, freq)
            is_valid, message = validate_training_data(ts)
            if not is_valid:
                st.error(message, icon="🚨")
                st.markdown(
                    """
                    <div class='error-box'>
                    <strong>Need help?</strong> Ensure your file includes columns: date, demand. 
                    Supported aliases include 'quantity' for 'demand'. 
                    <a href='#' onclick='st.session_state.show_template = True'>Download a sample template</a>.
                    </div>
                    """, unsafe_allow_html=True
                )
                if 'show_template' in st.session_state and st.session_state.show_template:
                    template_data = pd.DataFrame({
                        'date': ['2023-01-01', '2023-02-01'],
                        'demand': [100, 120]
                    })
                    st.download_button(
                        label="Download Sample CSV",
                        data=template_data.to_csv(index=False),
                        file_name="sample_demand_data.csv",
                        mime="text/csv"
                    )
                return None
            return ts
    except Exception as e:
        st.error(
            f"Error preparing time series: {str(e)}. Please check the data format and column names.", 
            icon="🚨"
        )
        return None

def train_xgboost_model(ts: pd.Series, test_size: float, config: Dict, use_cache: bool) -> Tuple[Optional[Dict], Optional[pd.Index], Optional[pd.Series]]:
    """
    Train an XGBoost model and evaluate it.

    Args:
        ts (pd.Series): Time series data.
        test_size (float): Test size percentage (0-1).
        config (Dict): Model configuration for fingerprinting.
        use_cache (bool): Whether to use model caching.

    Returns:
        Tuple[Optional[Dict], Optional[pd.Index], Optional[pd.Series]]: Model data, actual index, actual values.
    """
    try:
        features_df = ForecastEngine.create_features(ts.to_frame(), ts.name)
        if features_df is None or features_df.empty:
            st.error(
                "Failed to create features for XGBoost. Ensure the data has sufficient records.", 
                icon="🚨"
            )
            return None, None, None

        X = features_df.drop(columns=[ts.name, 'date'])
        y = features_df[ts.name]

        if len(X) < 2:
            st.error("Insufficient data for training after feature creation.")
            return None, None, None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        with st.spinner("Training XGBoost model..."):
            model = ModelTrainer.train_xgboost(X_train, y_train, use_cache=use_cache, config=config)
            if model is None:
                st.error("XGBoost model training failed.", icon="🚨")
                return None, None, None

            evaluation = ModelTrainer.evaluate_model(model, X_test, y_test)
            last_values = X.iloc[-1].values

            return {
                'model': model,
                'last_values': last_values,
                'evaluation': evaluation
            }, y_test.index, y_test.values
    except Exception as e:
        st.error(
            f"Error training XGBoost model: {str(e)}. Please verify the data and parameters.", 
            icon="🚨"
        )
        return None, None, None

def train_arima_model(ts: pd.Series, test_size: float) -> Tuple[Optional[Dict], Optional[pd.Index], Optional[pd.Series]]:
    """
    Train an ARIMA model and evaluate it.

    Args:
        ts (pd.Series): Time series data.
        test_size (float): Test size percentage (0-1).

    Returns:
        Tuple[Optional[Dict], Optional[pd.Index], Optional[pd.Series]]: Model data, actual index, actual values.
    """
    try:
        train_size = int(len(ts) * (1 - test_size))
        if train_size < 1 or len(ts) - train_size < 1:
            st.error("Insufficient data for train-test split.", icon="🚨")
            return None, None, None

        train, test = ts[:train_size], ts[train_size:]

        with st.spinner("Training ARIMA model..."):
            model = ModelTrainer.train_arima(train)
            if model is None:
                st.error("ARIMA model training failed.", icon="🚨")
                return None, None, None

            evaluation = ModelTrainer.evaluate_model(model, test.index, test.values)
            return {
                'model': model,
                'evaluation': evaluation
            }, test.index, test.values
    except Exception as e:
        st.error(
            f"Error training ARIMA model: {str(e)}. Please verify the data and parameters.", 
            icon="🚨"
        )
        return None, None, None

def render_training_ui(mode: str) -> Optional[Dict]:
    """
    Render the model training UI based on the mode.

    Args:
        mode (str): Application mode ('Simple' or 'Technical').

    Returns:
        Optional[Dict]: Training configuration or None if invalid.
    """
    st.header("Model Training", help="Train a forecasting model to predict future demand.")

    if mode == "Simple":
        st.markdown(
            "**Simple Mode: Train with Default Settings**", 
            help="Select a model to train with default parameters."
        )
        model_type = st.selectbox(
            "Select Model Type",
            list(MODEL_TYPES.keys()),
            help="Choose a model for forecasting (e.g., XGBoost for complex patterns, ARIMA for time series)."
        )
        forecast_period = DEFAULT_FORECAST_PERIOD
        test_size = 0.2
        use_cache = True
    else:
        st.markdown(
            "**Technical Mode: Customize Training**", 
            help="Configure model type, forecast period, and test size."
        )
        model_type = st.selectbox(
            "Select Model Type",
            list(MODEL_TYPES.keys()),
            help="Choose a model for forecasting (e.g., XGBoost for complex patterns, ARIMA for time series)."
        )
        forecast_period = st.number_input(
            "Forecast Period (months)",
            min_value=1, max_value=24, value=DEFAULT_FORECAST_PERIOD,
            help="Number of months to forecast (1-24).",
            key="training_forecast_period_input"
        )
        test_size = st.slider(
            "Test Size Percentage",
            10, 40, 20,
            help="Percentage of data to use for testing (10-40%)."
        )
        use_cache = st.checkbox(
            "Use model caching",
            value=True,
            help="Cache the trained model to avoid retraining with the same data and settings."
        )

    return {
        "model_type": model_type,
        "forecast_period": forecast_period,
        "test_size": test_size / 100,
        "use_cache": use_cache
    }

def show_model_training() -> None:
    """
    Render the UI for training forecasting models and display results.

    Supports Simple Mode (default settings) and Technical Mode (custom settings).
    """
    # Check session state
    if 'state' not in st.session_state or not isinstance(st.session_state.state, SessionState):
        st.error("Session state not initialized. Please restart the app.", icon="🚨")
        return

    # Check data availability
    if st.session_state.state.data is None:
        st.warning(
            "No data loaded. Please configure and load data in the 'Data Source' section.", 
            icon="⚠️"
        )
        st.button("Go to Data Source", on_click=lambda: st.session_state.update(page="data_source"))
        return

    df = st.session_state.state.data
    mode = st.session_state.get('mode', 'Simple')

    # Prepare time series data
    ts = prepare_training_data(df)
    if ts is None:
        return

    # Render UI and get configuration
    config = render_training_ui(mode)
    if not config:
        return

    # Train model button
    if st.button("🚀 Train Model", help="Train the selected model with the specified settings."):
        model_type = config["model_type"]
        try:
            if model_type == "XGBoost":
                model_data, actual_index, actual_values = train_xgboost_model(
                    ts, config["test_size"], config, config["use_cache"]
                )
            elif model_type == "ARIMA":
                model_data, actual_index, actual_values = train_arima_model(
                    ts, config["test_size"]
                )
            else:
                st.error(
                    f"Unsupported model type: {model_type}. Please select a valid model.", 
                    icon="🚨"
                )
                return

            if model_data is None:
                return

            # Store model in session state
            st.session_state.state.models[model_type] = model_data
            st.success(f"{model_type} model trained successfully!", icon="✅")
            st.toast(f"✅ {model_type} model trained!")

            # Display evaluation results
            st.subheader(
                "Model Evaluation", 
                help="Metrics showing the model's performance on test data."
            )
            evaluation = model_data['evaluation']
            st.write(f"**Mean Absolute Error (MAE):** {evaluation['mae']:.2f} (average prediction error)")
            st.write(f"**Root Mean Squared Error (RMSE):** {evaluation['rmse']:.2f} (square root of average squared errors)")

            # Plot predictions
            with st.spinner("Generating prediction plot..."):
                plot_predictions(actual_index, actual_values, evaluation['predictions'])
        except Exception as e:
            st.error(
                f"Unexpected error during training: {str(e)}. Please check the data and settings.", 
                icon="🚨"
            )
            st.toast("❌ Training failed.")