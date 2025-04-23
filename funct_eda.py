import streamlit as st
from cls_data_preprocessor import DataProcessor
from cls_plots_visuals import Visualizer


def show_data_exploration():
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