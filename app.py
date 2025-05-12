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

# Custom CSS for modern, consistent styling
st.markdown("""
    <style>
    .main { padding: 2rem; background-color: #f9fafb; }
    .stButton > button { 
        width: 100%; 
        padding: 10px; 
        font-size: 16px; 
        border-radius: 8px; 
        background-color: #2563eb; 
        color: white; 
        border: none; 
    }
    .stButton > button:hover { background-color: #1e40af; }
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
    </style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state with defaults
def initialize_session_state():
    if 'state' not in st.session_state:
        logger.info("Initializing SessionState")
        st.session_state.state = SessionState()
        st.session_state.state.data_source = "csv"
        st.session_state.state.data = None
        st.session_state.state.processed_data = None
        st.session_state.state.models = {}  # Ensure models dict exists

    # These should be initialized outside the above block to ensure they always exist
    if 'mode' not in st.session_state:
        st.session_state.mode = "Simple"

    if 'model_params' not in st.session_state:
        st.session_state.model_params = {"forecast_horizon": 12}

    if 'forecast_generated' not in st.session_state:
        st.session_state.forecast_generated = False

    if 'current_forecast' not in st.session_state:
        st.session_state.current_forecast = None


# Cache data loading with hashable inputs
@st.cache_data(show_spinner=False)
def cached_load_data(_data_source: str, _file=None, _connection_string: str = "", _api_config: dict = {}) -> Optional[pd.DataFrame]:
    """Load and validate data with caching"""
    try:
        data = load_data()
        if data is None:
            return None
        required_cols = [
            STANDARD_COLUMNS['date'],
            STANDARD_COLUMNS['demand'],
            STANDARD_COLUMNS['delivery_date'],
            STANDARD_COLUMNS['delivery_quantity']
        ]
        is_valid, message = DataProcessor.validate_columns(data, required_cols)
        if not is_valid:
            st.error(message, icon="🚨")
            st.markdown(
                """
                <div class='error-box'>
                <strong>Need help?</strong> Ensure your file includes columns: date, demand, delivery_date, delivery_quantity. 
                Supported aliases include 'quantity' for 'demand'. 
                <a href='#' onclick='st.session_state.show_template = True'>Download a sample template</a>.
                </div>
                """, unsafe_allow_html=True
            )
            if 'show_template' in st.session_state and st.session_state.show_template:
                template_data = pd.DataFrame({
                    'date': ['2023-01-01', '2023-02-01'],
                    'demand': [100, 120],
                    'delivery_date': ['2023-01-05', '2023-02-06'],
                    'delivery_quantity': [95, 115]
                })
                st.download_button(
                    label="Download Sample CSV",
                    data=template_data.to_csv(index=False),
                    file_name="sample_demand_data.csv",
                    mime="text/csv"
                )
            return None
        return data
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        st.error(f"Failed to load data: {str(e)}. Please check file format or connection details.", icon="🚨")
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
        st.error("Time series data is invalid. Ensure the data contains valid date and demand columns.", icon="🚨")
        return False
    return True

# Render dashboard header
def render_dashboard_header():
    st.title("Product Demand Analysis Toolkit")
    st.markdown(
        """
        Analyze demand trends, generate forecasts, and optimize inventory. 
        Upload a CSV/Excel file to get started.
        """
    )
    st.markdown(
        '<div class="tooltip">ℹ️<span class="tooltiptext">Switch between Simple Mode for quick insights or Technical Mode for advanced analysis.</span></div>',
        unsafe_allow_html=True
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

        show_data_source_selection()

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
                        st.error("Data preprocessing failed. Check column requirements and file format.", icon="🚨")
                else:
                    st.error("Failed to load data. Ensure the file is CSV/Excel with required columns.", icon="🚨")

# Render Simple Mode dashboard
def show_simple_mode():
    st.header("Demand Planning Dashboard")
    st.markdown("Explore key insights and forecasts to optimize inventory.")

    # Two-column layout for overview and metrics
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.expander("Demand Forecast", expanded=True):
            show_forecasting()
            st.markdown(
                '<div class="tooltip">ℹ️<span class="tooltiptext">Use these forecasts to adjust inventory levels in high-demand periods.</span></div>',
                unsafe_allow_html=True
            )
            st.markdown("**Recommendation**: Increase stock in high-demand regions based on trends.")

    with col2:
        with st.expander("Key Metrics", expanded=True):
            data = st.session_state.state.data
            st.metric(
                "Average Demand",
                f"{data[STANDARD_COLUMNS['demand']].mean():.2f}",
                help="Average demand across all periods."
            )
            ts = DataProcessor.prepare_time_series(data)
            anomalies_count = len(detect_sales_anomalies(ts)) if validate_time_series(ts) else 0
            st.metric(
                "Recent Anomalies",
                anomalies_count,
                help="Number of unusual demand spikes or drops."
            )

    # Visualizations section
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

    # Download button
    st.download_button(
        label="Download Insights",
        data=st.session_state.state.data.to_csv(index=False),
        file_name="demand_insights.csv",
        mime="text/csv",
        help="Download a summary of demand data and insights."
    )

# Render Technical Mode dashboard
def show_technical_mode():
    st.header("Technical Analysis Dashboard")
    st.markdown("Dive into detailed data analysis, model tuning, and diagnostics.")

    # Tabbed interface for technical features
    tabs = st.tabs([
        "Data Exploration",
        "Model Tuning",
        "Forecasting",
        "Regional Analysis",
        "Anomaly Detection",
        "Discontinued Products",
        "Model Management"
    ])

    with tabs[0]:
        show_data_exploration()

    with tabs[1]:
        st.subheader("Model Parameters")
        forecast_horizon = st.slider(
            "Forecast Horizon (months)",
            1,
            24,
            st.session_state.model_params["forecast_horizon"],
            help="Set the number of months to forecast."
        )
        st.session_state.model_params["forecast_horizon"] = forecast_horizon
        show_model_training()

    with tabs[2]:
        st.subheader("Demand Forecasts")
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

    with tabs[5]:
        st.subheader("Discontinued Products")
        threshold = st.slider(
            "Months without orders",
            2,
            12,
            3,
            help="Set the threshold for identifying discontinued products."
        )
        discontinued = detect_discontinued_products(st.session_state.state.data, threshold)
        if STANDARD_COLUMNS['material'] in st.session_state.state.data.columns:
            st.dataframe(discontinued)
            fig = px.bar(
                discontinued.head(20),
                x='material',
                y='Months_Since_Last_Order',
                title="Top 20 Discontinued Products",
                color='Potentially_Discontinued'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "Material column not found. Ensure the dataset includes a 'Material' or equivalent column.",
                icon="⚠️"
            )

    with tabs[6]:
        show_model_management()

# Main App
def main():
    initialize_session_state()
    render_dashboard_header()
    render_sidebar()

    # Check if data is loaded
    if st.session_state.state.data is None:
        st.info(
            """
            Please upload a CSV/Excel file or connect to a data source to begin analysis. 
            Required columns: date, demand, delivery_date, delivery_quantity.
            """,
            icon="ℹ️"
        )
        return

    # Render mode-specific UI
    if st.session_state.mode == "Simple":
        show_simple_mode()
    else:
        show_technical_mode()

if __name__ == "__main__":
    main()