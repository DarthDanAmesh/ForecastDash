import streamlit as st
import plotly.express as px
from cls_session_management import SessionState
from cls_data_preprocessor import DataProcessor
from typing import List, Optional
import pandas as pd
import logging
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

# Configuration
st.set_page_config(page_title="Product Demand Toolkit", layout="wide", initial_sidebar_state="expanded")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'state' not in st.session_state:
    logger.info("Initializing new SessionState")
    st.session_state.state = SessionState()
    # Initialize data_source attribute
    st.session_state.state.data_source = "csv"  # Default data source
    # Initialize data attribute
    st.session_state.state.data = None
    
if 'mode' not in st.session_state:
    logger.info("Setting default mode to Simple")
    st.session_state.mode = "Simple"  # Default to Simple Mode

# Cache data loading to optimize performance
@st.cache_data
def cached_load_data():
    """Cached wrapper for load_data function"""
    return load_data()

# Cache feature engineering
@st.cache_data
def cached_feature_engineering(data):
    """Cached wrapper for enhance_feature_engineering function"""
    logger.info(f"Calling cached_feature_engineering with data shape: {data.shape if data is not None else 'None'}")
    if data is None:
        return None
    return enhance_feature_engineering(data)

# Helper function to check data validity
def is_data_valid(data: Optional[pd.DataFrame]) -> bool:
    """Check if the data is valid for analysis"""
    if data is None:
        return False
    if len(data) == 0:
        return False
    return True



# Main App
def main():
    st.title("Product Demand Analysis and Prediction Toolkit")

    # Sidebar for mode selection and data upload
    with st.sidebar:
        st.header("Control Panel")
        # Mode toggle
        mode = st.radio(
            "Select Mode",
            ["Simple", "Technical"],
            index=0 if st.session_state.mode == "Simple" else 1,
            help="Simple Mode: View key insights and forecasts. Technical Mode: Access detailed analysis and model tuning."
        )
        st.session_state.mode = mode

        # Data source selection
        show_data_source_selection()

        # Load data button with enhanced error handling
        if st.button("Load Data", help="Upload and process your demand data"):
            with st.spinner("Loading data..."):
                try:
                    data = cached_load_data()
                    if data is not None:
                        st.session_state.state.data = cached_feature_engineering(data)
                        st.success("Data loaded successfully!")
                    else:
                        st.error("Failed to load data. Please check the file format (CSV/Excel) and ensure it contains required columns (e.g., date, demand).")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}. Please verify the file and try again.")

    # UI based on mode
    if st.session_state.mode == "Simple":
        show_simple_mode()
    else:
        show_technical_mode()

# Simple Mode: Streamlined dashboard for non-technical users
def show_simple_mode():
    if st.session_state.state.data is None:
        st.warning("No data loaded. Please upload data in the sidebar.")
        return

    st.header("Demand Planning Dashboard")
    st.markdown("View key insights, forecasts, and recommendations for inventory management.")

    # Key Insights Section
    with st.expander("Key Insights", expanded=True):
        st.subheader("Forecast Summary")
        show_forecasting()
        st.subheader("Recommended Actions")
        st.write("- Adjust inventory for high-demand regions based on regional analysis.")
        st.write("- Review potentially discontinued products for phase-out.")

    # Visualizations Section
    with st.expander("Visualizations", expanded=True):
        # Regional Performance
        st.subheader("Regional Performance")
        region_performance = analyze_regional_performance(st.session_state.state.data)
        fig1, fig2 = plot_regional_performance(region_performance)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        # Anomalies
        st.subheader("Sales Anomalies")
        ts = DataProcessor.prepare_time_series(st.session_state.state.data)
        anomalies = detect_sales_anomalies(ts)
        st.plotly_chart(plot_anomalies(anomalies), use_container_width=True)

    # Downloadable Summary
    st.download_button(
        label="Download Summary Report",
        data=st.session_state.state.data.to_csv(index=False),
        file_name="demand_summary.csv",
        mime="text/csv",
        help="Download a CSV with key demand insights."
    )

# Technical Mode: Detailed analysis for data scientists
def show_technical_mode():
    if st.session_state.state.data is None:
        st.warning("No data loaded. Please upload data in the sidebar.")
        return

    st.header("Technical Analysis and Model Management")
    st.markdown("Explore data, tune models, and analyze detailed metrics.")

    # Data Exploration
    with st.expander("Data Exploration", expanded=False):
        show_data_exploration()

    # Model Training and Tuning
    with st.expander("Model Training", expanded=False):
        show_model_training()
        st.subheader("Parameter Tuning")
        # Example: Add sliders for model hyperparameters (to be implemented in model_training.py)
        st.slider("Forecast Horizon (months)", 1, 24, 12, key="forecast_horizon")

    # Forecasting (Standard and Extended)
    with st.expander("Forecasting", expanded=False):
        st.subheader("Standard Forecasting")
        show_forecasting()
        st.subheader("Extended Forecasting (18 Months)")
        show_extended_forecasting()

    # Regional Analysis
    with st.expander("Regional Analysis", expanded=False):
        region_performance = analyze_regional_performance(st.session_state.state.data)
        fig1, fig2 = plot_regional_performance(region_performance)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(region_performance)

    # Anomaly Detection
    with st.expander("Anomaly Detection", expanded=False):
        ts = DataProcessor.prepare_time_series(st.session_state.state.data)
        anomalies = detect_sales_anomalies(ts)
        st.plotly_chart(plot_anomalies(anomalies), use_container_width=True)
        st.subheader("Detected Anomalies")
        anomaly_table = anomalies[anomalies['is_anomaly']].reset_index()
        anomaly_table.columns = ['Date', 'Value', 'Rolling Mean', 'Upper Bound', 'Lower Bound', 'Is Anomaly']
        st.dataframe(anomaly_table)

    # Discontinued Products
    with st.expander("Discontinued Products", expanded=False):
        threshold = st.slider("Months without orders to consider discontinued", 2, 12, 3)
        discontinued = detect_discontinued_products(st.session_state.state.data, threshold)
        st.dataframe(discontinued)
        fig = px.bar(discontinued.head(20), x='Material', y='Months_Since_Last_Order',
                     title="Top 20 Products by Months Since Last Order",
                     color='Potentially_Discontinued')
        st.plotly_chart(fig, use_container_width=True)

    # Model Management
    with st.expander("Model Management", expanded=False):
        show_model_management()



if __name__ == "__main__":
    main()