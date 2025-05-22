# funct_model_training.py
import streamlit as st
import pandas as pd
from typing import Optional, Dict
from cls_data_preprocessor import DataProcessor
from forecast_engine import ForecastEngine
from cls_session_management import SessionState
from constants import DEFAULT_PREDICTION_LENGTH, STANDARD_COLUMNS

def validate_training_data(df: pd.DataFrame, min_length: int = 12) -> tuple[bool, str]:
    """Validate DataFrame for model training."""
    if df is None or df.empty:
        return False, "Data is empty. Ensure the dataset includes 'date', 'demand', and 'material' columns."
    if not all(col in df.columns for col in [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['material']]):
        return False, "Missing required columns: date, demand, material."
    if len(df) < min_length:
        return False, f"Insufficient data: At least {min_length} rows required, got {len(df)}."
    if df[STANDARD_COLUMNS['demand']].isna().any():
        return False, "Demand column contains missing values. Please preprocess the data."
    return True, ""

@st.cache_data(show_spinner=False)
def prepare_training_data(_df: pd.DataFrame, target_col: str = STANDARD_COLUMNS['demand'], 
                         date_col: str = STANDARD_COLUMNS['date']) -> Optional[pd.DataFrame]:
    """Prepare DataFrame for model training."""
    try:
        if _df is None or _df.empty:
            st.error("No data provided. Please upload a dataset.", icon="🚨")
            return None

        with st.spinner("Preparing data..."):
            df = DataProcessor.preprocess_data(_df)
            is_valid, message = validate_training_data(df)
            if not is_valid:
                st.error(message, icon="🚨")
                st.markdown(
                    """
                    <div class='error-box'>
                    <strong>Need help?</strong> Ensure your file includes columns: date, demand, material.
                    <a href='#' onclick='st.session_state.show_template = True'>Download a sample template</a>.
                    </div>
                    """, unsafe_allow_html=True
                )
                if 'show_template' in st.session_state and st.session_state.show_template:
                    template_data = pd.DataFrame({
                        'date': ['2023-01-01', '2023-02-01'],
                        'demand': [100, 120],
                        'material': ['SKU001', 'SKU001']
                    })
                    st.download_button(
                        label="Download Sample CSV",
                        data=template_data.to_csv(index=False),
                        file_name="sample_demand_data.csv",
                        mime="text/csv"
                    )
                return None
            return df
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}.", icon="🚨")
        return None

def render_training_ui(mode: str) -> Optional[Dict]:
    """Render model training UI based on mode."""
    st.header("Model Training", help="Generate forecasts for demand planning.")

    if mode == "Simple":
        st.markdown("**Simple Mode: Default Forecast**", help="Generate forecasts with default settings.")
        forecast_period = DEFAULT_PREDICTION_LENGTH
    else:
        st.markdown("**Technical Mode: Customize Forecast**", help="Configure forecast horizon.")
        forecast_period = st.number_input(
            "Forecast Period (weeks)",
            min_value=1, max_value=24, value=DEFAULT_PREDICTION_LENGTH,
            help="Number of weeks to forecast (1-24).",
            key="training_forecast_period_input"
        )

    return {"forecast_period": forecast_period}

def show_model_training() -> None:
    """Render UI for generating forecasts."""
    if 'state' not in st.session_state or not isinstance(st.session_state.state, SessionState):
        st.error("Session state not initialized. Please restart the app.", icon="🚨")
        return

    if st.session_state.state.data is None:
        st.warning("No data loaded. Please configure and load data in the 'Data Source' section.", icon="⚠️")
        return

    df = st.session_state.state.data
    mode = st.session_state.get('mode', 'Simple')

    df = prepare_training_data(df)
    if df is None:
        return

    config = render_training_ui(mode)
    if not config:
        return

    if st.button("🚀 Generate Forecast", help="Generate demand forecasts."):
        with st.spinner("Generating forecast..."):
            try:
                forecast = ForecastEngine.forecast(df, forecast_horizon=config["forecast_period"])
                if forecast is None:
                    st.error("Failed to generate forecast.", icon="🚨")
                    return

                st.session_state.state.forecasts["DeepAR"] = forecast
                st.session_state.state.models["DeepAR"] = {"trained_at": pd.Timestamp.now()}
                st.success("Forecast generated successfully!", icon="✅")
                st.toast("✅ Forecast ready!")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}.", icon="🚨")
                st.toast("❌ Forecast failed.")