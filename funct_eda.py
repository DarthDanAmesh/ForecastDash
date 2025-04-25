import streamlit as st
from cls_data_preprocessor import DataProcessor
from cls_plots_visuals import Visualizer
from consts_model import COUNTRY_CODE_MAP

from typing import List
from pygwalker.api.streamlit import init_streamlit_comm, StreamlitRenderer, PreFilter
import pandas as pd

def show_data_exploration():
    st.header("Data Exploration")
    
    if st.session_state.state.data is None:
        st.warning("No data loaded. Please configure and load data first.")
        return
    
    df = st.session_state.state.data
    
    # Initialize pygwalker communication
    init_streamlit_comm()
    
    # Show raw data option
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)
    
    # Data summary
    st.subheader("Data Summary")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Date Range: {df['Created On'].min()} to {df['Created On'].max()}")
    
    # Traditional plots section for quick insights
    with st.expander("Quick Insights"):
        # Time series plot
        ts = DataProcessor.prepare_time_series(df)
        if ts is not None:
            st.plotly_chart(Visualizer.plot_time_series(ts, "Delivery Quantity Over Time"))
        
        # Geographical distribution
        if 'Country Key Ship-to' in df.columns:
            st.plotly_chart(Visualizer.plot_geographical(df))
        
        # Product performance
        if 'Material Group' in df.columns:
            st.plotly_chart(Visualizer.plot_product_performance(df))
    
    # Advanced Interactive Visualization with pygwalker
    st.subheader("Interactive Data Exploration")
    
    # Cache the pygwalker renderer
    @st.cache_resource
    def get_pyg_renderer():
        return StreamlitRenderer(df, debug=False)
    
    renderer = get_pyg_renderer()
    
    # Create filters for key dimensions
    filter_cols = []
    pre_filters = []
    
    # Add Country filter if available
        # Filter by country full names instead of codes
    if 'Country Key Ship-to' in df.columns:
        available_codes = df['Country Key Ship-to'].dropna().unique().tolist()
        code_to_name = {code: COUNTRY_CODE_MAP.get(code, code) for code in available_codes}
        name_to_code = {v: k for k, v in code_to_name.items()}

        country_labels = list(code_to_name.values())
        selected_labels = st.multiselect("Filter by Country", country_labels, [])

        selected_codes = [name_to_code[label] for label in selected_labels if label in name_to_code]

        if selected_codes:
            pre_filters.append(PreFilter(
                field="Country Key Ship-to",
                op="one of",
                value=selected_codes
            ))
    
    # Add Material Group filter if available
    if 'Material Group' in df.columns:
        material_groups = df['Material Group'].unique().tolist()
        selected_material_groups = st.multiselect('Filter by Material Group', material_groups, [])
        if selected_material_groups:
            pre_filters.append(PreFilter(
                field="Material Group",
                op="one of",
                value=selected_material_groups
            ))
    
    # Add date range filter
    if 'Created On' in df.columns:
        min_date = df['Created On'].min().date()
        max_date = df['Created On'].max().date()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)
        
        if start_date and end_date:
            pre_filters.append(PreFilter(
                field="Created On",
                op="temporal range",
                value=[str(start_date), str(end_date)]
            ))
    
    # Apply filters
    renderer.set_global_pre_filters(pre_filters)
    
    # Create tabs for different analysis perspectives
    geo_tab, product_tab, procurement_tab = st.tabs([
        "Geographical Analysis", 
        "Product Performance", 
        "Procurement Analysis"
    ])
    
    # Populate tabs with relevant visualizations
    # Note: You'll need to create and save these chart configurations
    # using renderer.render_explore() first, then use the chart indices
    
    with geo_tab:
        st.subheader("Geographical Distribution")
        # Assuming chart index 0 is geographical visualization
        try:
            renderer.chart(0)  # Distribution by country
        except:
            st.info("Create geographical charts first using the Explore UI")
            
        # You could add a second visualization here
        # renderer.chart(1)  # Another geographical insight
    
    with product_tab:
        st.subheader("Product Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Assuming chart indices 2 and 3 are product visualizations
            try:
                renderer.chart(2)  # Top products by quantity
            except:
                st.info("Create product performance charts first using the Explore UI")
        with col2:
            try:
                renderer.chart(3)  # Top products by value
            except:
                pass
    
    with procurement_tab:
        st.subheader("Procurement Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Assuming chart indices 6 and 7 are procurement visualizations
            try:
                renderer.chart(6)  # Analysis by procurement type
            except:
                st.info("Create procurement analysis charts first using the Explore UI")
        with col2:
            try:
                renderer.chart(7)  # Analysis by production source
            except:
                pass
    
    # Option to explore with the full pygwalker UI
    if st.checkbox("Advanced Chart Builder (for analysts)"):
        st.subheader("Custom Chart Builder")
        renderer.explorer()


#--------------------------------------------------------------------
#deprecated   
#--------------------------------------------------------------------
#deprecated        
def show_data_exploration_depr():
    st.header("Data Exploration")
    
    if st.session_state.state.data is None:
        st.warning("No data loaded. Please configure and load data first.")
        return
    
    df = st.session_state.state.data
    
    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)
    
    # Data summary
    st.subheader("Data Summary")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Date Range: {df['Created On'].min()} to {df['Created On'].max()}")
    
    # Time series plot
    ts = DataProcessor.prepare_time_series(df)
    if ts is not None:
        st.plotly_chart(Visualizer.plot_time_series(ts, "Delivery Quantity Over Time"))
    
    # Geographical distribution
    if 'Country Key Ship-to' in df.columns:
        st.plotly_chart(Visualizer.plot_geographical(df))
    
    # Product performance
    if 'Material Group' in df.columns:
        st.plotly_chart(Visualizer.plot_product_performance(df))


