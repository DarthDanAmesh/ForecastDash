# funct_eda.py
import streamlit as st
import pandas as pd
from cls_data_preprocessor import DataProcessor
from funct_abnormal_detect import detect_sales_anomalies, plot_anomalies
from constants import STANDARD_COLUMNS, COUNTRY_CODE_MAP
import plotly.express as px
from pygwalker.api.streamlit import init_streamlit_comm, StreamlitRenderer, PreFilter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_data_exploration():
    """Render data exploration UI with SKU-level insights."""
    st.header("Data Exploration")
    
    if st.session_state.state.data is None:
        st.warning("No data loaded. Please configure and load data first.", icon="⚠️")
        return
    
    df = st.session_state.state.data
    if not all(col in df.columns for col in [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand']]):
        st.error("Data is missing required columns: date, demand.", icon="🚨")
        return
    
    init_streamlit_comm()
    
    if st.checkbox("Show Raw Data"):
        st.dataframe(df, use_container_width=True)
    
    st.subheader("Data Summary")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Date Range: {df[STANDARD_COLUMNS['date']].min()} to {df[STANDARD_COLUMNS['date']].max()}")
    st.write(f"Unique SKUs: {df[STANDARD_COLUMNS['material']].nunique()}")
    
    st.subheader("Quick Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Use raw DataFrame instead of prepare_time_series
            if STANDARD_COLUMNS['material'] in df.columns:
                # Aggregate demand by date and material
                plot_data = df.groupby([STANDARD_COLUMNS['date'], STANDARD_COLUMNS['material']])[STANDARD_COLUMNS['demand']].sum().reset_index()
                fig = px.line(
                    plot_data,
                    x=STANDARD_COLUMNS['date'],
                    y=STANDARD_COLUMNS['demand'],
                    color=STANDARD_COLUMNS['material'],
                    title="Demand Over Time by Material",
                    labels={'x': 'Date', 'y': 'Demand', STANDARD_COLUMNS['material']: 'Material'}
                )
            else:
                # Aggregate demand by date only
                plot_data = df.groupby(STANDARD_COLUMNS['date'])[STANDARD_COLUMNS['demand']].sum().reset_index()
                fig = px.line(
                    plot_data,
                    x=STANDARD_COLUMNS['date'],
                    y=STANDARD_COLUMNS['demand'],
                    title="Total Demand Over Time",
                    labels={'x': 'Date', 'y': 'Demand'}
                )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            st.error(f"Failed to create time series plot: {str(e)}.", icon="🚨")
    
    with col2:
        if STANDARD_COLUMNS['country'] in df.columns:
            try:
                country_summary = df.groupby(STANDARD_COLUMNS['country'])[STANDARD_COLUMNS['demand']].sum().reset_index()
                fig = px.bar(
                    country_summary,
                    x=STANDARD_COLUMNS['country'],
                    y=STANDARD_COLUMNS['demand'],
                    title="Demand by Country",
                    labels={STANDARD_COLUMNS['demand']: 'Total Demand'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Error creating country bar plot: {str(e)}")
                st.error(f"Failed to create country bar plot: {str(e)}.", icon="🚨")
    
    if STANDARD_COLUMNS['material'] in df.columns:
        try:
            sku_summary = df.groupby(STANDARD_COLUMNS['material'])[STANDARD_COLUMNS['demand']].sum().nlargest(5).reset_index()
            fig = px.bar(
                sku_summary,
                x=STANDARD_COLUMNS['material'],
                y=STANDARD_COLUMNS['demand'],
                title="Top 5 SKUs by Demand",
                labels={STANDARD_COLUMNS['demand']: 'Total Demand'}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating SKU bar plot: {str(e)}")
            st.error(f"Failed to create SKU bar plot: {str(e)}.", icon="🚨")
    
    st.subheader("Anomaly Insights")
    ts = DataProcessor.prepare_time_series(df)
    if ts is not None and not ts.empty:
        try:
            anomalies = detect_sales_anomalies(ts)
            if anomalies is not None:
                fig = plot_anomalies(anomalies, title="Total Demand Anomaly Detection")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No anomalies detected.", icon="⚠️")
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            st.error(f"Failed to perform anomaly detection: {str(e)}.", icon="🚨")
    
    st.subheader("Interactive Data Exploration")
    
    @st.cache_resource
    def get_pyg_renderer():
        return StreamlitRenderer(df, debug=False)
    
    renderer = get_pyg_renderer()
    
    st.subheader("Filters")
    pre_filters = []
    
    if STANDARD_COLUMNS['country'] in df.columns:
        available_codes = df[STANDARD_COLUMNS['country']].dropna().unique().tolist()
        code_to_name = {code: COUNTRY_CODE_MAP.get(code, code) for code in available_codes}
        name_to_code = {v: k for k, v in code_to_name.items()}
        country_labels = list(code_to_name.values())
        selected_labels = st.multiselect("Filter by Country", country_labels, [])
        selected_codes = [name_to_code[label] for label in selected_labels if label in name_to_code]
        if selected_codes:
            pre_filters.append(PreFilter(field=STANDARD_COLUMNS['country'], op='one of', value=selected_codes))
    
    if STANDARD_COLUMNS['material'] in df.columns:
        materials = df[STANDARD_COLUMNS['material']].dropna().unique().tolist()
        selected_materials = st.multiselect("Filter by Material", materials, [])
        if selected_materials:
            pre_filters.append(PreFilter(field=STANDARD_COLUMNS['material'], op='one of', value=selected_materials))
    
    if STANDARD_COLUMNS['date'] in df.columns:
        min_date = df[STANDARD_COLUMNS['date']].min().date()
        max_date = df[STANDARD_COLUMNS['date']].max().date()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)
        if start_date and end_date:
            pre_filters.append(PreFilter(field=STANDARD_COLUMNS['date'], op='temporal range', value=[str(start_date), str(end_date)]))
    
    renderer.set_global_pre_filters(pre_filters)
    
    geo_tab, product_tab = st.tabs(["Geographical Analysis", "Product Performance"])
    
    with geo_tab:
        st.subheader("Geographical Distribution")
        # In the geo_tab section
        try:
            renderer.chart(0)
        except Exception as e:
            logger.error(f"Error rendering geographical chart: {str(e)}")
            st.info("Create geographical charts first using the Explore UI", icon="ℹ️")
        
        # In the product_tab section
        try:
            renderer.chart(1)
        except Exception as e:
            logger.error(f"Error rendering product chart: {str(e)}")
            st.info("Create product performance charts first using the Explore UI", icon="ℹ️")