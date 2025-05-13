# simple_mode_ui.py
import streamlit as st
import pandas as pd
from typing import Optional
from cls_data_preprocessor import DataProcessor
from cls_forecast_engine import ForecastEngine
from simple_mode_visualizations import plot_historical_forecast_comparison, display_kpis
from simple_mode_insights import display_top_skus, display_discontinued_products, display_recommendation_cards
from simple_mode_utils import filter_data, calculate_performance_gaps
from funct_shw_forecast_plot import cached_generate_forecast, update_forecast_state, display_forecast
from consts_model import DEFAULT_FORECAST_PERIOD
from column_config import STANDARD_COLUMNS
import logging

logger = logging.getLogger(__name__)

def show_simple_mode():
    """Render Simple Mode UI for non-technical users."""
    if st.session_state.state.data is None:
        st.error("No data loaded. Please upload a CSV/Excel file in the sidebar.", icon="🚨")
        return

    # Layout: Sidebar, Main Content, Right Panel
    col_main, col_right = st.columns([4, 1])
    with st.sidebar:
        render_filters()
    with col_main:
        render_central_controls()
        render_data_table()
        render_forecast_section()
    with col_right:
        render_sku_panel()

def render_filters():
    """Render sidebar filters for material and country."""
    st.markdown("<div class='sidebar-filter'>", unsafe_allow_html=True)
    st.subheader("Filters")
    
    data = st.session_state.state.data
    material_col = STANDARD_COLUMNS['material']
    country_col = STANDARD_COLUMNS['country']
    
    material_options = ['All'] + sorted(data[material_col].unique().tolist()) if material_col in data.columns else ['All']
    country_options = ['All'] + sorted(data[country_col].unique().tolist()) if country_col in data.columns else ['All']
    
    selected_materials = st.multiselect(
        "Select SKUs/Materials",
        material_options,
        default=['All'],
        help="Filter by product SKU or material code.",
        key="material_filter"
    )
    selected_countries = st.multiselect(
        "Select Countries",
        country_options,
        default=['All'],
        help="Filter by country or location.",
        key="country_filter"
    )
    
    if 'All' in selected_materials:
        selected_materials = material_options[1:]
    if 'All' in selected_countries:
        selected_countries = country_options[1:]
    
    st.session_state.filters = {
        'materials': selected_materials,
        'countries': selected_countries
    }
    st.markdown("</div>", unsafe_allow_html=True)

def render_central_controls():
    """Render date range selector, forecast freeze, and method selection."""
    st.subheader("Demand Planning Controls")
    data = st.session_state.state.data
    date_col = STANDARD_COLUMNS['date']
    
    if data is None or date_col not in data.columns:
        st.error("Dataset missing date column. Please upload a valid file.", icon="🚨")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            min_date = pd.to_datetime(data[date_col]).min()
            max_date = pd.to_datetime(data[date_col]).max()
            st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                help="Choose the time period for analysis.",
                key="date_range"
            )
        except Exception as e:
            st.error(f"Invalid date column: {str(e)}. Ensure dates are in a valid format.", icon="🚨")
            return
    
    with col2:
        st.checkbox(
            "Freeze Forecast",
            value=False,
            help="Lock the forecast to prevent updates until unfrozen.",
            key="freeze_forecast"
        )
    
    with col3:
        st.selectbox(
            "Forecast Method",
            ["ARIMA"],
            disabled=True,
            help="Default method for simple forecasting. Technical Mode offers more options.",
            key="forecast_method"
        )

def render_data_table():
    """Render monthly demand table with performance gaps."""
    st.subheader("Monthly Demand Overview")
    data = filter_data(st.session_state.state.data)
    if data.empty:
        st.warning("No data matches the selected filters.", icon="⚠️")
        return
    
    ts = DataProcessor.prepare_time_series(data)
    if ts is None or ts.empty:
        st.error("Unable to process data. Ensure date and demand columns are valid.", icon="🚨")
        return
    
    monthly_data = ts.resample('ME').sum().reset_index()
    monthly_data['Performance Gap'] = calculate_performance_gaps(ts)
    
    styled_data = monthly_data.style.apply(
        lambda row: ['background-color: #ffcccc' if row['Performance Gap'] < 0 else '' for _ in row],
        axis=1
    ).format({
        STANDARD_COLUMNS['demand']: "{:.2f}",
        'Performance Gap': "{:.2f}%"
    })
    
    st.dataframe(styled_data, use_container_width=True)
    st.markdown(
        '<div class="tooltip">ℹ️<span class="tooltiptext">Red highlights indicate months with demand below expectations.</span></div>',
        unsafe_allow_html=True
    )

def render_forecast_section():
    """Render forecast generation and visualization section."""
    st.subheader("Demand Forecast")
    if st.session_state.get('freeze_forecast', False):
        st.warning("Forecast is frozen. Uncheck 'Freeze Forecast' to generate a new forecast.", icon="⚠️")
        return
    
    if st.session_state.forecast_generated and st.session_state.current_forecast is not None:
        ts = DataProcessor.prepare_time_series(st.session_state.state.data)
        display_forecast(ts, st.session_state.current_forecast)
        plot_historical_forecast_comparison(ts, st.session_state.current_forecast)
        display_kpis()
        display_top_skus()
        display_discontinued_products()
        display_recommendation_cards()
        return
    
    if st.button("Generate Forecast", type="primary", help="Generate a demand forecast"):
        with st.spinner("Generating forecast..."):
            ts = DataProcessor.prepare_time_series(st.session_state.state.data)
            if ts is None or ts.empty:
                st.error("Invalid data. Ensure date and demand columns are valid.", icon="🚨")
                return
            try:
                model_info = ForecastEngine.train_arima(ts)
                st.session_state.state.models["ARIMA"] = model_info
                forecast = cached_generate_forecast(ts, "ARIMA", DEFAULT_FORECAST_PERIOD, model_info)
                if forecast is not None:
                    update_forecast_state("ARIMA", DEFAULT_FORECAST_PERIOD, forecast)
                    display_forecast(ts, forecast)
                    plot_historical_forecast_comparison(ts, forecast)
                    display_kpis()
                    display_top_skus()
                    display_discontinued_products()
                    display_recommendation_cards()
                    st.success("Forecast generated successfully!", icon="✅")
                else:
                    st.error("Failed to generate forecast. Check data quality.", icon="🚨")
            except Exception as e:
                logger.error(f"Forecast error: {str(e)}")
                st.error(f"Failed to generate forecast: {str(e)}. Contact support.", icon="🚨")

def render_sku_panel():
    """Render right-side panel with selected SKUs and quantities."""
    st.markdown("<div class='sidebar-filter'>", unsafe_allow_html=True)
    st.subheader("Selected SKUs")
    data = filter_data(st.session_state.state.data)
    if data.empty:
        st.info("No SKUs selected. Use sidebar filters.", icon="ℹ️")
        return
    
    material_col = STANDARD_COLUMNS['material']
    if material_col not in data.columns:
        st.warning("Material column missing.", icon="⚠️")
        return
    
    sku_summary = data.groupby(material_col)[STANDARD_COLUMNS['demand']].sum().reset_index()
    sku_summary.columns = ['SKU', 'Total Demand']
    
    st.dataframe(
        sku_summary.style.format({'Total Demand': "{:.2f}"}),
        use_container_width=True
    )
    st.button("Adjust Stock", help="Initiate stock adjustment for selected SKUs.", key="adjust_stock")
    st.markdown("</div>", unsafe_allow_html=True)