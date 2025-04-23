import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

def load_data_from_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def load_data_from_database(connection_string, query=None):
    try:
        engine = create_engine(connection_string)
        if query is None:
            query = "SELECT * FROM products"
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

def load_data_from_api(api_config):
    # Placeholder for API implementation
    try:
        # In a real implementation, you would make API calls here
        st.warning("API data loading not implemented in this example")
        return None
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return None