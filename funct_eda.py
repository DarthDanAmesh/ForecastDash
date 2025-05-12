import streamlit as st
from cls_data_preprocessor import DataProcessor
from cls_plots_visuals import Visualizer
from consts_model import COUNTRY_CODE_MAP
from pygwalker.api.streamlit import init_streamlit_comm, StreamlitRenderer, PreFilter
import pandas as pd
from cls_session_management import SessionState

#state = SessionState.get_or_create()

def show_data_exploration():
    st.header("Data Exploration")
    
    if st.session_state.state.data is None:
        st.warning("No data loaded. Please configure and load data first.")
        return
    
    df = st.session_state.state.data

    # Use standardized column names
    if 'country' not in df.columns or 'date' not in df.columns:
        st.error("Data is missing required standardized columns like 'country' or 'date'.")
        return
    
    # Initialize pygwalker communication
    init_streamlit_comm()
    
    # Show raw data option
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)
    
    # Data summary
    st.subheader("Data Summary")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    
    # Quick insights section
    st.subheader("Quick Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        ts = DataProcessor.prepare_time_series(df)
        if ts is not None:
            st.plotly_chart(Visualizer.plot_time_series(ts, "Demand Over Time"), use_container_width=True)
    
    with col2:
        if 'country' in df.columns:
            st.plotly_chart(Visualizer.plot_geographical(df), use_container_width=True)
    
    if 'material' in df.columns:
        st.plotly_chart(Visualizer.plot_product_performance(df), use_container_width=True)
    
    # Interactive section
    st.subheader("Interactive Data Exploration")
    
    @st.cache_resource
    def get_pyg_renderer():
        return StreamlitRenderer(df, debug=False)
    
    renderer = get_pyg_renderer()
    
    st.subheader("Filters")
    pre_filters = []

    if 'country' in df.columns:
        available_codes = df['country'].dropna().unique().tolist()
        code_to_name = {code: COUNTRY_CODE_MAP.get(code, code) for code in available_codes}
        name_to_code = {v: k for k, v in code_to_name.items()}
        country_labels = list(code_to_name.values())
        selected_labels = st.multiselect("Filter by Country", country_labels, [])
        selected_codes = [name_to_code[label] for label in selected_labels if label in name_to_code]
        if selected_codes:
            pre_filters.append(PreFilter(field='country', op='one of', value=selected_codes))

    if 'material' in df.columns:
        materials = df['material'].dropna().unique().tolist()
        selected_materials = st.multiselect("Filter by Material", materials, [])
        if selected_materials:
            pre_filters.append(PreFilter(field='material', op='one of', value=selected_materials))

    if 'date' in df.columns:
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)
        if start_date and end_date:
            pre_filters.append(PreFilter(field='date', op='temporal range', value=[str(start_date), str(end_date)]))

    renderer.set_global_pre_filters(pre_filters)

    geo_tab, product_tab, procurement_tab = st.tabs([
        "Geographical Analysis", 
        "Product Performance", 
        "Procurement Analysis"
    ])
    
    with geo_tab:
        st.subheader("Geographical Distribution")
        try:
            renderer.chart(0)
        except:
            st.info("Create geographical charts first using the Explore UI")
    
    with product_tab:
        st.subheader("Product Analysis")
        col1, col2 = st.columns(2)
        with col1:
            try:
                renderer.chart(2)
            except:
                st.info("Create product performance charts first using the Explore UI")
        with col2:
            try:
                renderer.chart(3)
            except:
                pass
    
    with procurement_tab:
        st.subheader("Procurement Analysis")
        col1, col2 = st.columns(2)
        with col1:
            try:
                renderer.chart(6)
            except:
                st.info("Create procurement analysis charts first using the Explore UI")
        with col2:
            try:
                renderer.chart(7)
            except:
                pass
    
    if st.checkbox("Advanced Chart Builder (for analysts)"):
        st.subheader("Custom Chart Builder")
        renderer.explorer()
