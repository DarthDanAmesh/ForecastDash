# funct_load_data.py
import streamlit as st
import pandas as pd
import tempfile
from typing import Optional
import logging
from cls_data_preprocessor import DataProcessor
from constants import STANDARD_COLUMNS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_error(message: str, exception: Exception = None) -> None:
    """Centralized error handling and logging."""
    st.error(message, icon="🚨")
    logger.error(f"{message}{f': {str(exception)}' if exception else ''}")

@st.cache_data(show_spinner=False)
def load_and_preprocess_csv(_file_path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess data from a CSV file."""
    try:
        logger.info(f"Loading file: {_file_path}")
        df = pd.read_csv(_file_path)
        if df.empty:
            handle_error("Uploaded file is empty.")
            return None
        with st.spinner("Preprocessing data..."):
            df = DataProcessor.preprocess_data(df)
            return df
    except Exception as e:
        handle_error(f"Error loading CSV file: {str(e)}", e)
        return None

@st.cache_data(show_spinner=False)
def load_and_preprocess_api(_api_config: dict) -> Optional[pd.DataFrame]:
    """Load and preprocess data from an API."""
    try:
        logger.info("Loading data from API")
        from source_loading_util_funct import load_data_from_api
        df = load_data_from_api(_api_config)
        if df.empty:
            handle_error("API returned no data.")
            return None
        with st.spinner("Preprocessing data..."):
            df = DataProcessor.preprocess_data(df)
            return df
    except Exception as e:
        handle_error(f"Error loading API data: {str(e)}", e)
        return None

@st.cache_data(show_spinner=False)
def load_and_preprocess_database(_connection_string: str, query: str) -> Optional[pd.DataFrame]:
    """Load and preprocess data from a database."""
    try:
        logger.info(f"Loading data from database with query: {query}")
        from source_loading_util_funct import load_data_from_database
        df = load_data_from_database(_connection_string, query)
        if df.empty:
            handle_error("Database query returned no data.")
            return None
        with st.spinner("Preprocessing data..."):
            df = DataProcessor.preprocess_data(df)
            return df
    except Exception as e:
        handle_error(f"Error loading database data: {str(e)}", e)
        return None

@st.cache_data(show_spinner=False)
def load_data(data_source: str, uploaded_file=None, connection_string: str = "", api_config: dict = {}) -> Optional[pd.DataFrame]:
    """Load and preprocess data based on the specified data source."""
    logger.info(f"Starting load_data function with data_source: {data_source}")
    
    # Initialize session_state.state if not present
    if 'state' not in st.session_state:
        st.session_state.state = type('State', (), {})()
    
    with st.spinner("Loading data..."):
        if data_source == "csv":
            if uploaded_file is None:
                handle_error("No file uploaded for CSV data source.")
                return None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            data = load_and_preprocess_csv(tmp_file_path)
        elif data_source == "api":
            if not api_config:
                handle_error("API configuration is missing.")
                return None
            data = load_and_preprocess_api(api_config)
        elif data_source == "database":
            if not connection_string:
                handle_error("Database connection string is missing.")
                return None
            query = "SELECT * FROM demand_data"  # Adjust based on your needs
            data = load_and_preprocess_database(connection_string, query)
        else:
            handle_error(f"Unsupported data source: {data_source}")
            return None
        
        if data is not None:
            st.session_state.state.data = data
            logger.info(f"Data loaded successfully with {len(data)} rows")
            with st.expander("📊 Preview of Preprocessed Data", expanded=False):
                st.subheader("Preview of Preprocessed Data")
                st.dataframe(data.head(5))
                st.write(f"Total rows: {len(data)}")
                st.write(f"Columns: {list(data.columns)}")
        return data