# app.py
import streamlit as st
import logging
import pandas as pd
from typing import Optional
from cls_session_management import SessionState
from ui_data_source import show_data_source_selection
from simple_mode_ui import show_simple_mode
from funct_eda import show_data_exploration
from funct_model_training import show_model_training
from funct_shw_forecast_plot import show_forecasting
from ui_extended_forecast import show_extended_forecasting
from regional_perf_analysis import analyze_regional_performance, plot_regional_performance
from funct_abnormal_detect import detect_sales_anomalies, plot_anomalies
from funct_detect_prod_discontinued import detect_discontinued_products
from ui_model_management import show_model_management
from funct_load_data import load_data
from funct_feature_eng import enhance_feature_engineering
from cls_data_preprocessor import DataProcessor
from constants import STANDARD_COLUMNS
import plotly.express as px
import tempfile

# Configuration
st.set_page_config(page_title="Franke Demand Toolkit", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (unchanged)
st.markdown("""
    <style>
    .main { padding: 2rem; background-color: #f9fafb; }
    .stButton > button { 
        padding: 10px; 
        font-size: 16px; 
        border-radius: 8px; 
        background-color: #005670; 
        color: white; 
        border: none; 
    }
    .stButton > button:hover { background-color: #003d4f; }
    .st-expander { 
        border: 1px solid #e5e7eb; 
        border-radius: 8px; 
        margin-bottom: 1rem; 
        background-color: white; 
    }
    .tooltip { position: relative; display: inline-block; cursor: help; }
    .tooltip .tooltiptext { 
        visibility: hidden; 
        width: 220px; 
        background-color: #374151; 
        color: #fff; 
        text-align: center; 
        border-radius: 6px; 
        padding: 8px; 
        position: absolute; 
        z-index: 1; 
        bottom: 125%; 
        left: 50%; 
        margin-left: -110px; 
        opacity: 0; 
        transition: opacity 0.3s; 
    }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
    .error-box { 
        background-color: #fee2e2; 
        padding: 1rem; 
        border-radius: 8px; 
        border: 1px solid #ef4444; 
    }
    .nav-bar { 
        background-color: #005670; 
        padding: 1rem; 
        color: white; 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
    }
    .nav-bar h1 { margin: 0; font-size: 24px; }
    .sidebar-filter { background-color: #f1f5f9; padding: 1rem; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def initialize_session_state():
    if 'state' not in st.session_state or not isinstance(st.session_state.state, SessionState):
        logger.info("Initializing SessionState")
        st.session_state.state = SessionState()
        st.session_state.state.data_source = "csv"
        st.session_state.state.data = None
        st.session_state.state.processed_data = None
        st.session_state.state.models = {}
        st.session_state.state.forecasts = {}

    if 'mode' not in st.session_state:
        st.session_state.mode = "Simple"
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {"forecast_horizon": 2}  # Aligned with test.py
    if 'forecast_generated' not in st.session_state:
        st.session_state.forecast_generated = False
    if 'current_forecast' not in st.session_state:
        st.session_state.current_forecast = None

# Cache data loading
@st.cache_data(show_spinner=False)
def cached_load_data(_data_source: str, _uploaded_file=None, _connection_string: str = "", _api_config: dict = {}) -> Optional[pd.DataFrame]:
    try:
        data = load_data(_data_source, _uploaded_file, _connection_string, _api_config)
        if data is None:
            return None
        required_cols = [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand']]
        optional_cols = [
            STANDARD_COLUMNS['material'], 
            STANDARD_COLUMNS['country'], 
            STANDARD_COLUMNS['delivery_date'], 
            STANDARD_COLUMNS['delivery_quantity']
        ]
        is_valid, message = DataProcessor.validate_columns(data, required_cols, optional_cols)
        if not is_valid:
            st.error(message, icon="🚨")
            st.markdown(
                """
                <div class='error-box'>
                <strong>Need help?</strong> Ensure your file includes columns: date, demand. 
                Supported aliases: 'quantity' for demand, 'itemcode' for material.
                <a href='#' onclick='st.session_state.show_template = True'>Download sample template</a>.
                </div>
                """, unsafe_allow_html=True
            )
            if 'show_template' in st.session_state and st.session_state.show_template:
                template_data = pd.DataFrame({
                    'date': ['2023-01-01', '2023-02-01'],
                    'demand': [100, 120],
                    'material': ['ANA-1011', 'BAK-101'],
                    'country': ['US', 'US']
                })
                st.download_button(
                    label="Download Sample CSV",
                    data=template_data.to_csv(index=False),
                    file_name="sample_demand_data.csv",
                    mime="text/csv"
                )
            return None
        if message:
            st.warning(message, icon="⚠️")
        return data
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        st.error(f"Failed to load data: {str(e)}. Please check file format.", icon="🚨")
        return None

# Cache feature engineering
@st.cache_data(show_spinner=False)
def cached_feature_engineering(_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    if _data is None:
        return None
    return enhance_feature_engineering(_data)

# Render navigation bar
def render_nav_bar():
    st.markdown(
        """
        <div class='nav-bar'>
            <h1>Franke Demand Toolkit</h1>
            <div>
                <span class='tooltip'>ℹ️<span class='tooltiptext'>Contact support: support@franke.com</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# Render sidebar controls
def render_sidebar():
    with st.sidebar:
        st.header("Control Panel")
        mode = st.radio(
            "Select Mode",
            ["Simple", "Technical"],
            index=0 if st.session_state.mode == "Simple" else 1,
            help="Simple Mode for quick insights, Technical Mode for detailed analysis."
        )
        st.session_state.mode = mode

        show_data_source_selection() # This function should update st.session_state.state.uploaded_file

        # Conditionally display the "Load Data" button
        if st.session_state.state.data_source == "csv":
            if hasattr(st.session_state.state, "uploaded_file") and st.session_state.state.uploaded_file is not None:
                if st.button("Load Data", type="primary", help="Upload and process your demand data"):
                    with st.spinner("Loading and processing data..."):
                        data = cached_load_data(
                            st.session_state.state.data_source,
                            st.session_state.state.uploaded_file,
                            getattr(st.session_state.state, "connection_string", ""),
                            getattr(st.session_state.state, "api_config", {})
                        )
                        if data is not None:
                            processed_data = DataProcessor.preprocess_data(data)
                            if processed_data is not None:
                                st.session_state.state.data = cached_feature_engineering(processed_data)
                                st.success("Data loaded and processed successfully!", icon="✅")
                            else:
                                st.error(
                                    "Data preprocessing failed. Check column requirements and file format.", 
                                    icon="🚨"
                                )
                        else:
                            st.error(
                                "Failed to load data. Ensure the file is CSV/Excel with required columns.", 
                                icon="🚨"
                            )
            else:
                st.info("Please upload a CSV file to enable data loading.") # Optional: Inform user
        else: # For other data sources, show the button directly
            if st.button("Load Data", type="primary", help="Upload and process your demand data"):
                with st.spinner("Loading and processing data..."):
                    data = cached_load_data(
                        st.session_state.state.data_source,
                        getattr(st.session_state.state, "uploaded_file", None),
                        getattr(st.session_state.state, "connection_string", ""),
                        getattr(st.session_state.state, "api_config", {})
                    )
                    if data is not None:
                        processed_data = DataProcessor.preprocess_data(data)
                        if processed_data is not None:
                            st.session_state.state.data = cached_feature_engineering(processed_data)
                            st.success("Data loaded and processed successfully!", icon="✅")
                        else:
                            st.error(
                                "Data preprocessing failed. Check column requirements and file format.", 
                                icon="🚨"
                            )
                    else:
                        st.error(
                            "Failed to load data. Ensure the file is CSV/Excel with required columns.", 
                            icon="🚨"
                        )

# Technical Mode dashboard
def show_technical_mode():
    st.header("Technical Analysis Dashboard")
    tabs = st.tabs([
        "Data Exploration", "Model Tuning", "Forecasting",
        "Regional Analysis", "Anomaly Detection", "Discontinued Products", "Model Management"
    ])
    with tabs[0]:
        show_data_exploration()
    with tabs[1]:
        st.subheader("Model Parameters")
        forecast_horizon = st.slider(
            "Forecast Horizon (weeks)", 1, 12, st.session_state.model_params["forecast_horizon"],
            help="Set forecast duration (aligned with weekly data)."
        )
        st.session_state.model_params["forecast_horizon"] = forecast_horizon
        show_model_training()
    with tabs[2]:
        st.subheader("Demand Forecasts")
        show_forecasting()
        st.subheader("Extended Forecast")
        show_extended_forecasting()
    with tabs[3]:
        try:
            region_performance = analyze_regional_performance(st.session_state.state.data)
            fig1, fig2 = plot_regional_performance(region_performance)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(region_performance['region'])
        except Exception as e:
            st.error(f"Failed to display regional performance: {str(e)}", icon="🚨")
    with tabs[4]:
        if st.session_state.state.data is not None and not st.session_state.state.data.empty:
            try:
                # Convert wide-format DataFrame to long format
                data = st.session_state.state.data
                if all(col in data.columns for col in [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['material']]):
                    anomaly_data = data
                else:
                    # Assume wide format with date index and material columns
                    logger.info(f"Converting wide-format DataFrame to long format. Columns: {list(data.columns)}")
                    anomaly_data = data.reset_index().melt(
                        id_vars=[STANDARD_COLUMNS['date']] if STANDARD_COLUMNS['date'] in data.index.name else ['index'],
                        value_vars=[col for col in data.columns if col not in [STANDARD_COLUMNS['date'], 'index']],
                        var_name=STANDARD_COLUMNS['material'],
                        value_name=STANDARD_COLUMNS['demand']
                    )
                    if 'index' in anomaly_data.columns:
                        anomaly_data = anomaly_data.rename(columns={'index': STANDARD_COLUMNS['date']})
                    anomaly_data = anomaly_data.dropna(subset=[STANDARD_COLUMNS['demand']])
                    logger.info(f"Converted to long format. Columns: {list(anomaly_data.columns)}")
                
                anomalies = detect_sales_anomalies(anomaly_data)
                if anomalies is not None and not anomalies.empty:
                    st.plotly_chart(plot_anomalies(anomalies), use_container_width=True)
                    if 'is_anomaly' in anomalies.columns:
                        st.subheader("Detected Anomalies")
                        anomaly_table = anomalies[anomalies['is_anomaly']].reset_index()
                        anomaly_table.columns = ['Index', 'Material', 'Date', 'Demand', 'Rolling Mean', 'Upper Bound', 'Lower Bound', 'Is Anomaly']
                        st.dataframe(anomaly_table)
                    else:
                        st.warning("No anomalies detected.", icon="⚠️")
                else:
                    st.warning("No anomalies detected or data is invalid.", icon="⚠️")
            except Exception as e:
                st.error(f"Failed to perform anomaly detection: {str(e)}", icon="🚨")
                logger.error(f"Anomaly detection failed: {str(e)}", exc_info=True)
    with tabs[5]:
        st.subheader("Discontinued Products")
        threshold = st.slider("Months without orders", 2, 12, 3, help="Threshold for discontinuation.")
        discontinued = detect_discontinued_products(st.session_state.state.data, threshold)
        if STANDARD_COLUMNS['material'] in st.session_state.state.data.columns:
            st.dataframe(discontinued)
            fig = px.bar(
                discontinued.head(20), x='material', y='Months_Since_Last_Order',
                title="Top 20 Discontinued Products", color='Potentially_Discontinued'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Material column missing.", icon="⚠️")
    with tabs[6]:
        show_model_management()

# Main App
def main():
    initialize_session_state()
    render_nav_bar()
    render_sidebar()

    if st.session_state.state.data is None:
        st.info(
            "Upload a CSV file to start. Required columns: date, demand. Optional: material, country.",
            icon="ℹ️"
        )
        return

    if st.session_state.mode == "Simple":
        show_simple_mode()
    else:
        show_technical_mode()

if __name__ == "__main__":
    main()