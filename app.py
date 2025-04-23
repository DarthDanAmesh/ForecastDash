import streamlit as st


from cls_session_management import SessionState
from ui_data_source import show_data_source_selection
from ui_model_management import show_model_management
from funct_load_data import load_data
from funct_eda import show_data_exploration
from funct_model_training import show_model_training
from funct_shw_forecast_plot import show_forecasting

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
                st.sidebar.success("Data loaded successfully!")
            else:
                st.sidebar.error("Failed to load data")
    
    # Main tabs
    tabs = st.tabs(["Data Exploration", "Model Training", "Forecasting", "Model Management"])
    
    with tabs[0]:
        show_data_exploration()
    
    with tabs[1]:
        show_model_training()
    
    with tabs[2]:
        show_forecasting()
    
    with tabs[3]:
        show_model_management()

if __name__ == "__main__":
    main()