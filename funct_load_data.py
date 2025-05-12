# funct_load.py
import streamlit as st
import pandas as pd
from typing import Optional
import logging
from cls_data_preprocessor import DataProcessor
from source_loading_util_funct import load_data_from_api, load_data_from_csv, load_data_from_database
from column_config import standardize_column_names

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_inputs(data_source: str, session_state: object) -> tuple[bool, str]:
    """
    Validate inputs for the specified data source.
    """
    logger.info(f"Validating inputs for data source: {data_source}")
    if data_source == "csv":
        if not hasattr(session_state, "uploaded_file") or session_state.uploaded_file is None:
            return False, "No file uploaded. Please upload a CSV or Excel file."
        if not session_state.uploaded_file.name.endswith((".csv", ".xlsx")):
            return False, "Invalid file format. Please upload a CSV or Excel file."
    elif data_source == "database":
        if not hasattr(session_state, "connection_string") or not session_state.connection_string:
            return False, "No database connection string provided."
        if not isinstance(session_state.connection_string, str) or not session_state.connection_string.strip():
            return False, "Invalid database connection string."
    elif data_source == "api":
        if not hasattr(session_state, "api_config") or not session_state.api_config:
            return False, "No API configuration provided."
        if not isinstance(session_state.api_config, dict) or not session_state.api_config:
            return False, "Invalid API configuration. Must be a non-empty dictionary."
    else:
        return False, "Invalid data source selected. Choose 'csv', 'database', or 'api'."
    logger.info("Input validation passed")
    return True, ""

def handle_error(message: str, exception: Exception = None) -> None:
    """
    Centralized error handling and logging.
    """
    st.error(message)
    logger.error(f"{message}{f': {str(exception)}' if exception else ''}")

@st.cache_data(show_spinner=False)
def load_and_preprocess_csv(_file, _file_name: str) -> Optional[pd.DataFrame]:
    """
    Load and preprocess data from a CSV or Excel file.
    """
    try:
        logger.info(f"Loading file: {_file_name}")
        if _file_name.endswith(".csv"):
            df = pd.read_csv(_file)
        else:
            df = pd.read_excel(_file)
        
        if df.empty:
            handle_error("Uploaded file is empty.")
            return None
        
        # Standardize column names
        df = standardize_column_names(df)
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            df = DataProcessor.preprocess_data(df)
            return df
            
    except Exception as e:
        handle_error(f"Error loading CSV/Excel file: {str(e)}", e)
        return None

@st.cache_data(show_spinner=False)
def load_and_preprocess_database(_connection_string: str, query: str) -> Optional[pd.DataFrame]:
    """
    Load and preprocess data from a database.
    """
    try:
        logger.info(f"Loading data from database with query: {query}")
        df = load_data_from_database(_connection_string, query)
        
        if df.empty:
            handle_error("Database query returned no data.")
            return None
        
        # Standardize column names
        df = standardize_column_names(df)
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            df = DataProcessor.preprocess_data(df)
            return df
            
    except Exception as e:
        handle_error(f"Error loading database data: {str(e)}", e)
        return None

@st.cache_data(show_spinner=False)
def load_and_preprocess_api(_api_config: str) -> Optional[pd.DataFrame]:
    """
    Load and preprocess data from an API.
    """
    try:
        logger.info("Loading data from API")
        df = load_data_from_api(_api_config)
        
        if df.empty:
            handle_error("API returned no data.")
            return None
        
        # Standardize column names
        df = standardize_column_names(df)
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            df = DataProcessor.preprocess_data(df)
            return df
            
    except Exception as e:
        handle_error(f"Error loading API data: {str(e)}", e)
        return None

def load_data() -> Optional[pd.DataFrame]:
    """
    Load and preprocess data based on the selected data source.
    """
    logger.info("Starting load_data function")
    
    if not hasattr(st.session_state, "state") or not isinstance(st.session_state.state, object):
        handle_error("Session state not properly initialized. Please restart the app.")
        return None
        
    session_state = st.session_state.state
    
    if not hasattr(session_state, "data_source"):
        handle_error("No data source selected. Please select a data source in the control panel.")
        return None
        
    data_source = session_state.data_source
    logger.info(f"Using data source: {data_source}")
    
    # Validate inputs
    is_valid, error_message = validate_inputs(data_source, session_state)
    if not is_valid:
        handle_error(error_message)
        return None
        
    # Load and preprocess data
    with st.spinner("Loading data..."):
        data = None
        if data_source == "csv":
            data = load_and_preprocess_csv(session_state.uploaded_file, session_state.uploaded_file.name)
        elif data_source == "database":
            query = getattr(session_state, "query", "SELECT * FROM demand_data")
            data = load_and_preprocess_database(session_state.connection_string, query)
        elif data_source == "api":
            data = load_and_preprocess_api(str(session_state.api_config))  # Serialize for caching
        else:
            handle_error(f"Unsupported data source: {data_source}")
            return None
                
        if data is not None:
            session_state.data = data
            logger.info(f"Data loaded successfully with {len(data)} rows")
            # Show preview for all data sources
            st.subheader("Preview of Preprocessed Data")
            st.dataframe(data.head(5))
        return data