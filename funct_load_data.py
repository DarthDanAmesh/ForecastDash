import streamlit as st
from source_loading_util_funct import load_data_from_api, load_data_from_csv, load_data_from_database
from cls_data_preprocessor import DataProcessor

# use the imported utility functions
def load_data():
    if st.session_state.state.data_source == "csv" and st.session_state.state.uploaded_file:
        df = load_data_from_csv(st.session_state.state.uploaded_file)
        st.session_state.state.data = DataProcessor.preprocess_data(df)
        
    elif st.session_state.state.data_source == "database" and st.session_state.state.connection_string:
        query = getattr(st.session_state.state, 'query', "SELECT * FROM demand_data")
        df = load_data_from_database(st.session_state.state.connection_string, query)
        st.session_state.state.data = DataProcessor.preprocess_data(df)
        
    elif st.session_state.state.data_source == "api" and st.session_state.state.api_config:
        df = load_data_from_api(st.session_state.state.api_config)
        st.session_state.state.data = DataProcessor.preprocess_data(df)