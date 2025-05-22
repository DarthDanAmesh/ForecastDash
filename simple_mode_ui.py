# simple_mode_ui.py
import streamlit as st
import pandas as pd
from typing import Optional
from cls_data_preprocessor import DataProcessor
from forecast_engine import ForecastEngine
from funct_shw_forecast_plot import display_forecast
from constants import STANDARD_COLUMNS, DEFAULT_PREDICTION_LENGTH
import logging

logger = logging.getLogger(__name__)

def show_simple_mode():
    """Render Simple Mode UI for non-technical users."""
    if st.session_state.state.data is None:
        st.error("No data loaded. Please upload a CSV file in the sidebar.", icon="🚨")
        return

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
    """Render date range selector and forecast freeze."""
    st.subheader("Demand Planning Controls")
    data = st.session_state.state.data
    date_col = STANDARD_COLUMNS['date']
    
    if data is None or date_col not in data.columns:
        st.error("Dataset missing date column. Please upload a valid file.", icon="🚨")
        return
    
    col1, col2 = st.columns(2)
    
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
            st.error(f"Invalid date column: {str(e)}.", icon="🚨")
            return
    
    with col2:
        st.checkbox(
            "Freeze Forecast",
            value=False,
            help="Lock the forecast to prevent updates until unfrozen.",
            key="freeze_forecast"
        )

def render_data_table():
    """Render monthly demand table."""
    st.subheader("Monthly Demand Overview")
    data = filter_data(st.session_state.state.data)
    if data.empty:
        st.warning("No data matches the selected filters.", icon="⚠️")
        return
    
    monthly_data = data.groupby([pd.Grouper(key=STANDARD_COLUMNS['date'], freq='ME'), STANDARD_COLUMNS['material']])[STANDARD_COLUMNS['demand']].sum().reset_index()
    monthly_data[STANDARD_COLUMNS['date']] = monthly_data[STANDARD_COLUMNS['date']].dt.strftime('%Y-%m')
    
    st.dataframe(monthly_data, use_container_width=True)

def render_forecast_section():
    """Render forecast generation and visualization section."""
    st.subheader("Demand Forecast")
    if st.session_state.get('freeze_forecast', False):
        st.warning("Forecast is frozen. Uncheck 'Freeze Forecast' to generate a new forecast.", icon="⚠️")
        return
    
    if st.session_state.get('forecast_generated', False) and "DeepAR" in st.session_state.state.forecasts:
        display_forecast(st.session_state.state.data, st.session_state.state.forecasts["DeepAR"])
        return
    
    if st.button("Generate Forecast", type="primary", help="Generate a demand forecast"):
        with st.spinner("Generating forecast..."):
            data = filter_data(st.session_state.state.data)
            if data.empty:
                st.error("No data after filtering. Adjust filters.", icon="🚨")
                return
            try:
                forecast = ForecastEngine.forecast(data, forecast_horizon=DEFAULT_PREDICTION_LENGTH)
                if forecast is None:
                    st.error("Failed to generate forecast.", icon="🚨")
                    return
                st.session_state.state.forecasts["DeepAR"] = forecast
                st.session_state.forecast_generated = True
                display_forecast(data, forecast)
                st.success("Forecast generated successfully!", icon="✅")
            except Exception as e:
                logger.error(f"Forecast error: {str(e)}")
                st.error(f"Failed to generate forecast: {str(e)}.", icon="🚨")

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

def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply filters to data based on session state."""
    filters = st.session_state.get('filters', {'materials': [], 'countries': []})
    filtered_data = data.copy()
    
    if filters['materials']:
        filtered_data = filtered_data[filtered_data[STANDARD_COLUMNS['material']].isin(filters['materials'])]
    if filters['countries']:
        filtered_data = filtered_data[filtered_data[STANDARD_COLUMNS['country']].isin(filters['countries'])]
    
    return filtered_data