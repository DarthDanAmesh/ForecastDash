import streamlit as st
import plotly.express as px
import pandas as pd
import logging
from typing import Optional
from cls_session_management import SessionState
from cls_data_preprocessor import DataProcessor
from ui_data_source import show_data_source_selection
from ui_model_management import show_model_management
from ui_extended_forecast import show_extended_forecasting
from regional_perf_analysis import analyze_regional_performance, plot_regional_performance
from funct_abnormal_detect import detect_sales_anomalies, plot_anomalies
from funct_feature_eng import enhance_feature_engineering
from funct_load_data import load_data
from funct_eda import show_data_exploration
from funct_model_training import show_model_training
from funct_shw_forecast_plot import show_forecasting
from funct_detect_prod_discontinued import detect_discontinued_products
from column_config import STANDARD_COLUMNS

# Configuration
st.set_page_config(page_title="Product Demand Toolkit", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for improved UI
st.markdown("""
    <style>
    .main { padding: 20px; }
    .stButton > button { width: 100%; font-size: 16px; }
    .st-expander { border: 1px solid #e0e0e0; border-radius: 5px; margin-bottom: 10px; }
    .tooltip { position: relative; display: inline-block; cursor: help; }
    .tooltip .tooltiptext { 
        visibility: hidden; width: 200px; background-color: #555; color: #fff; 
        text-align: center; border-radius: 6px; padding: 5px; position: absolute; 
        z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; 
        opacity: 0; transition: opacity 0.3s; 
    }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
    </style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'state' not in st.session_state:
    logger.info("Initializing SessionState")
    st.session_state.state = SessionState()
    st.session_state.state.data_source = "csv"
    st.session_state.state.data = None
    st.session_state.mode = "Simple"
    st.session_state.model_params = {"forecast_horizon": 12}

# Cache data loading with hashable inputs
@st.cache_data(show_spinner=False)
def cached_load_data(_data_source: str) -> Optional[pd.DataFrame]:
    """Load and validate data with caching"""
    try:
        data = load_data()
        if data is None:
            return None
        required_cols = [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand'], 
                        STANDARD_COLUMNS['delivery_date'], STANDARD_COLUMNS['delivery_quantity']]
        is_valid, message = DataProcessor.validate_columns(data, required_cols)
        if not is_valid:
            st.error(message)
            return None
        return data
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        return None

# Cache feature engineering
@st.cache_data(show_spinner=False)
def cached_feature_engineering(_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Cached feature engineering"""
    if _data is None:
        return None
    return enhance_feature_engineering(_data)

# Validate time series
def validate_time_series(ts: Optional[pd.Series]) -> bool:
    """Check if time series is valid"""
    if ts is None or ts.empty:
        st.error("Time series data is invalid. Ensure the data contains valid date and demand columns.")
        return False
    return True

# Main App
def main():
    st.title("Product Demand Analysis Toolkit")
    st.markdown("Upload a CSV/Excel file to analyze demand, view forecasts, and optimize inventory.")

    # Sidebar
    with st.sidebar:
        st.header("Control Panel")
        st.markdown('<div class="tooltip">ℹ️<span class="tooltiptext">Use Simple Mode for quick insights or Technical Mode for advanced analysis.</span></div>', 
                    unsafe_allow_html=True)
        mode = st.radio("Mode", ["Simple", "Technical"], index=0 if st.session_state.mode == "Simple" else 1)
        st.session_state.mode = mode

        show_data_source_selection()

        if st.button("Load Data", type="primary", help="Upload and process your demand data"):
            with st.spinner("Loading data..."):
                data = cached_load_data(st.session_state.state.data_source)
                if data is not None:
                    processed_data = DataProcessor.preprocess_data(data)
                    if processed_data is not None:
                        st.session_state.state.data = cached_feature_engineering(processed_data)
                        st.success("Data loaded and processed successfully!")
                    else:
                        st.error("Data preprocessing failed. Check column requirements and file format.")
                else:
                    st.error("Failed to load data. Ensure the file is CSV/Excel with columns: date, demand, delivery_date, delivery_quantity.")

    # Render UI based on mode
    if st.session_state.state.data is None:
        st.info("Please upload data to begin analysis.")
        return

    if st.session_state.mode == "Simple":
        show_simple_mode()
    else:
        show_technical_mode()

# Simple Mode
def show_simple_mode():
    st.header("Demand Planning Dashboard")
    st.markdown("Explore key insights and forecasts for inventory optimization.")

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("Forecast Overview", expanded=True):
            show_forecasting()
            st.markdown("**Recommended Inventory**: Increase stock in high-demand regions based on forecast trends.")

    with col2:
        with st.expander("Key Metrics", expanded=True):
            st.metric("Average Demand", 
                     f"{st.session_state.state.data[STANDARD_COLUMNS['demand']].mean():.2f}",
                     help="Average demand across all periods.")
            st.metric("Recent Anomalies", 
                     len(detect_sales_anomalies(DataProcessor.prepare_time_series(st.session_state.state.data))),
                     help="Number of unusual demand spikes/drops.")

    with st.expander("Visualizations", expanded=True):
        st.subheader("Regional Performance")
        region_performance = analyze_regional_performance(st.session_state.state.data)
        fig1, fig2 = plot_regional_performance(region_performance)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Sales Anomalies")
        ts = DataProcessor.prepare_time_series(st.session_state.state.data)
        if validate_time_series(ts):
            anomalies = detect_sales_anomalies(ts)
            st.plotly_chart(plot_anomalies(anomalies), use_container_width=True)

    st.download_button(
        label="Download Insights",
        data=st.session_state.state.data.to_csv(index=False),
        file_name="demand_insights.csv",
        mime="text/csv",
        help="Download a summary of demand data and insights."
    )

# Technical Mode
def show_technical_mode():
    st.header("Technical Analysis Dashboard")
    st.markdown("Analyze data, tune models, and explore detailed metrics.")

    tabs = st.tabs(["Data Exploration", "Model Tuning", "Forecasting", "Regional Analysis", 
                    "Anomaly Detection", "Discontinued Products", "Model Management"])

    with tabs[0]:
        show_data_exploration()

    with tabs[1]:
        st.subheader("Model Parameters")
        forecast_horizon = st.slider("Forecast Horizon (months)", 1, 24, 
                                   st.session_state.model_params["forecast_horizon"])
        st.session_state.model_params["forecast_horizon"] = forecast_horizon
        show_model_training()

    with tabs[2]:
        st.subheader("Forecasts")
        show_forecasting()
        st.subheader("Extended Forecast (18 Months)")
        show_extended_forecasting()

    with tabs[3]:
        region_performance = analyze_regional_performance(st.session_state.state.data)
        fig1, fig2 = plot_regional_performance(region_performance)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(region_performance)

    with tabs[4]:
        ts = DataProcessor.prepare_time_series(st.session_state.state.data)
        if validate_time_series(ts):
            anomalies = detect_sales_anomalies(ts)
            st.plotly_chart(plot_anomalies(anomalies), use_container_width=True)
            if not anomalies.empty and 'is_anomaly' in anomalies.columns:
                st.subheader("Detected Anomalies")
                anomaly_table = anomalies[anomalies['is_anomaly']].reset_index()
                anomaly_table.columns = ['Date', 'Value', 'Rolling Mean', 'Upper Bound', 
                                       'Lower Bound', 'Is Anomaly']
                st.dataframe(anomaly_table)
            else:
                st.warning("No anomalies detected or anomaly data is incomplete.")

    # In show_technical_mode, under "Discontinued Products" tab
    with tabs[5]:
        threshold = st.slider("Months without orders", 2, 12, 3)
        discontinued = detect_discontinued_products(st.session_state.state.data, threshold)
        if STANDARD_COLUMNS['material'] in st.session_state.state.data.columns:
            st.dataframe(discontinued)
            fig = px.bar(discontinued.head(20), x='Material', y='Months_Since_Last_Order',
                        title="Top 20 Discontinued Products",
                        color='Potentially_Discontinued')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Material column not found in the data. Please ensure the dataset includes a 'Material' or equivalent column.")

    with tabs[6]:
        show_model_management()

if __name__ == "__main__":
    main()