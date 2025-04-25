import streamlit as st
import plotly.express as px

from cls_session_management import SessionState
from cls_data_preprocessor import DataProcessor

from typing import List
import pandas as pd
from pygwalker.api.streamlit import init_streamlit_comm, StreamlitRenderer, PreFilter
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
st.set_page_config(page_title="Product Demand Toolkit", layout="wide")

#defines the app state
if 'state' not in st.session_state:
    st.session_state.state = SessionState()


# Main App
def main():
    st.title("Product Demand Analysis and Prediction Toolkit")
    
    # Data source selection
    show_data_source_selection()
    
    # Load data button
    if st.sidebar.button("Load Data"):
        with st.spinner("Loading data..."):
            load_data()
            if st.session_state.state.data is not None:
                # Apply enhanced feature engineering
                st.session_state.state.data = enhance_feature_engineering(st.session_state.state.data)
                st.sidebar.success("Data loaded successfully!")
            else:
                st.sidebar.error("Failed to load data")
    
    # Main tabs
    tabs = st.tabs([
        "Data Exploration", 
        "Model Training", 
        "Forecasting", 
        "Extended Forecasting (18 Months)",
        "Regional Analysis",
        "Anomaly Detection",
        "Discontinued Products",
        "Model Management",
        "Test"
    ])
    
    with tabs[0]:
        show_data_exploration()
    
    with tabs[1]:
        show_model_training()
    
    with tabs[2]:
        show_forecasting()
        
    with tabs[3]:
        show_extended_forecasting()
        
    with tabs[4]:
        if st.session_state.state.data is not None:
            region_performance = analyze_regional_performance(st.session_state.state.data)
            fig1, fig2 = plot_regional_performance(region_performance)
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
            st.dataframe(region_performance)
        else:
            st.warning("No data loaded. Please load data first.")
    
    with tabs[5]:
        if st.session_state.state.data is not None:
            st.header("Sales Anomaly Detection")
            ts = DataProcessor.prepare_time_series(st.session_state.state.data)
            anomalies = detect_sales_anomalies(ts)
            print("anoms",anomalies)
            st.plotly_chart(plot_anomalies(anomalies))
            
            # Show anomalies table
            st.subheader("Detected Anomalies")
            anomaly_table = anomalies[anomalies['is_anomaly']].reset_index()
            anomaly_table.columns = ['Date', 'Value', 'Rolling Mean', 'Upper Bound', 'Lower Bound', 'Is Anomaly']
            st.dataframe(anomaly_table)
        else:
            st.warning("No data loaded. Please load data first.")
    
    with tabs[6]:
        if st.session_state.state.data is not None:
            st.header("Potentially Discontinued Products")
            threshold = st.slider("Months without orders to consider discontinued", 2, 12, 3)
            discontinued = detect_discontinued_products(st.session_state.state.data, threshold)
            
            # Show discontinued products
            st.dataframe(discontinued)
            
            # Chart showing months since last order
            fig = px.bar(discontinued.head(20), x='Material', y='Months_Since_Last_Order',
                       title="Top 20 Products by Months Since Last Order",
                       color='Potentially_Discontinued')
            st.plotly_chart(fig)
        else:
            st.warning("No data loaded. Please load data first.")
    
    with tabs[7]:
        show_model_management()

    with tabs[8]:
        
        # Initialize pygwalker communication
        init_streamlit_comm()


        @st.cache_data
        def get_data() -> pd.DataFrame:
            return pd.read_csv(r"C:\Users\danie\Downloads\Billionaires Statistics Dataset.csv")


        @st.cache_data
        def get_all_country() -> List[str]:
            return get_data()["country"].unique().tolist()


        # You should cache your pygwalker renderer, if you don't want your memory to explode
        @st.cache_resource
        def get_pyg_renderer() -> "StreamlitRenderer":
            df = get_data()
            # When you need to publish your application, you need set `debug=False`,prevent other users to write your config file.
            return StreamlitRenderer(df, spec=r"C:\Users\danie\Downloads\billion_config.json", debug=False)


        renderer = get_pyg_renderer()

        # display explore ui, Developers can use this to prepare the charts you need to display.
        # renderer.render_explore()

        pre_filters = []

        selected_country = st.multiselect(
            'please select country',
            get_all_country(),
            []
        )

        if selected_country:
            pre_filters.append(PreFilter(
                field="country",
                op="one of",
                value=selected_country
            ))

        renderer.set_global_pre_filters(pre_filters)

        tab1, tab2 = st.tabs(
            ["Area Distribution", "Gender Distribution"]
        )

        # display chart ui
        with tab1:
            st.subheader("Country Distribution")
            renderer.chart(0)
            st.subheader("Area Distribution")
            renderer.chart(2)

        with tab2:
            st.subheader("Gender Distribution")
            renderer.chart(1)
            st.subheader("Gender Distribution By Rank")
            renderer.chart(3)
            st.subheader("Gender Distribution By Age")
            renderer.chart(4)


if __name__ == "__main__":
    main()