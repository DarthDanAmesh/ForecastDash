# funct_load_data.py
import streamlit as st
import pandas as pd
from typing import Optional, Tuple
import logging
from cls_data_preprocessor import DataProcessor
from source_loading_util_funct import load_data_from_api, load_data_from_csv, load_data_from_database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_inputs(data_source: str, session_state: object) -> tuple[bool, str]:
    """
    Validate inputs for the specified data source.
    Args:
        data_source (str): The selected data source ('csv', 'database', 'api').
        session_state (object): The session state object containing input attributes.
    Returns:
        tuple[bool, str]: (is_valid, error_message) indicating if inputs are valid and any error message.
    """
    logger.info(f"Validating inputs for data source: {data_source}")
    if data_source == "csv":
        if not hasattr(session_state, "uploaded_file") or session_state.uploaded_file is None:
            return False, "No file uploaded. Please upload a CSV or Excel file."
        if not session_state.uploaded_file.name.endswith((".csv", ".xlsx")):
            return False, "Invalid file format. Please upload a CSV or Excel file."
    elif data_source == "database":
        if not hasattr(session_state, "connection_string") or not session_state.connection_string:
            return False, "No database connection string provided. Please enter a valid connection string."
    elif data_source == "api":
        if not hasattr(session_state, "api_config") or not session_state.api_config:
            return False, "No API configuration provided. Please provide valid API credentials."
    else:
        return False, "Invalid data source selected. Please choose 'csv', 'database', or 'api'."
    logger.info("Input validation passed")
    return True, ""

@st.cache_data
def load_and_preprocess_csv(_file) -> Optional[pd.DataFrame]:
    """
    Load and preprocess data from a CSV or Excel file.
    Args:
        _file: The uploaded file object (hashable for caching).
    Returns:
        Optional[pd.DataFrame]: The preprocessed DataFrame or None if loading fails.
    """
    try:
        logger.info(f"Loading file: {_file.name}")
        if _file.name.endswith(".csv"):
            df = pd.read_csv(_file)
        else:
            df = pd.read_excel(_file)
            
        # Log columns for debugging
        logger.info(f"Loaded columns: {df.columns.tolist()}")
        
        # Validate required columns with case-insensitive matching
        # To handle both 'date' and 'Date' columns
        required_columns = ["date", "demand"]
        available_columns = [col.lower() for col in df.columns]
        
        # Try to standardize column names (lowercase)
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        # Handle common column name variations
        column_mappings = {
            'created on': 'date',
            'created_on': 'date',
            'timestamp': 'date',
            'order date': 'date',
            'value': 'demand',
            'quantity': 'demand',
            'delivery quantity': 'demand',
            'sales': 'demand',
            'orders': 'demand',
            'country': 'country key ship-to',  # Add these common country column mappings
            'country_code': 'country key ship-to',
            'ship_to_country': 'country key ship-to',
            'customer country': 'country key ship-to'
        }
        
        # Apply mappings to standardize columns
        df = df.rename(columns=column_mappings)
        
        # Check for missing columns again
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}. Please ensure the file contains these columns or rename your columns to match: {required_columns}")
            # Show column help
            st.info(f"Your file has these columns: {', '.join(df.columns.tolist())}")
            st.info(f"We tried to map common alternatives like 'created on' -> 'date', 'quantity' -> 'demand', etc.")
            return None
            
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            try:
                # Convert date column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    # Check if conversion produced NaT values
                    if df['date'].isna().any():
                        st.warning(f"Some date values couldn't be parsed. {df['date'].isna().sum()} rows have invalid dates.")
                        # Drop rows with invalid dates
                        original_rows = len(df)
                        df = df.dropna(subset=['date'])
                        st.info(f"Dropped {original_rows - len(df)} rows with invalid dates.")
                        
                # Ensure demand is numeric
                if not pd.api.types.is_numeric_dtype(df['demand']):
                    try:
                        original_demand = df['demand'].copy()
                        df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
                        
                        # Report any conversion issues
                        if df['demand'].isna().any():
                            invalid_rows = df['demand'].isna().sum()
                            st.warning(f"{invalid_rows} rows had non-numeric demand values that were converted to NaN and will be removed.")
                            df = df.dropna(subset=['demand'])
                            
                        # Warn about zero values
                        zero_demand = (df['demand'] == 0).sum()
                        if zero_demand > 0:
                            st.warning(f"Found {zero_demand} rows with zero demand values (not removed)")
                            
                        # Warn about negative values
                        neg_demand = (df['demand'] < 0).sum()
                        if neg_demand > 0:
                            st.warning(f"Found {neg_demand} rows with negative demand values")
                            if st.checkbox("Convert negative demand to positive?"):
                                df['demand'] = df['demand'].abs()
                                
                    except Exception as e:
                        st.error(f"Error converting demand values to numbers: {str(e)}")
                        
                # Use DataProcessor for further preprocessing
                df = DataProcessor.preprocess_data(df)
                
                # Show a preview of the preprocessed data
                st.subheader("Preview of Preprocessed Data")
                st.dataframe(df.head(5))
                
                # Check if country column exists after mapping
                if 'country key ship-to' not in df.columns:
                    st.warning("No country column found - regional analysis will be disabled")
                    df['country key ship-to'] = 'Unknown'  # Add default value
                    
                return df
            
                problematic = df[df[STANDARD_COLUMNS['demand']].isna()]
                if not problematic.empty:
                    st.warning(f"Found {len(problematic)} rows with invalid demand values:")
                    st.dataframe(problematic[[STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand']]])
                
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
                logger.exception("Preprocessing error")
                return None
                
    except Exception as e:
        st.error(f"Error loading CSV/Excel file: {str(e)}. Please verify the file format and content.")
        logger.exception("File loading error")
        return None

@st.cache_data
def load_and_preprocess_database(_connection_string: str, query: str) -> Optional[pd.DataFrame]:
    """
    Load and preprocess data from a database.
    Args:
        _connection_string (str): Database connection string (hashable for caching).
        query (str): SQL query to execute.
    Returns:
        Optional[pd.DataFrame]: The preprocessed DataFrame or None if loading fails.
    """
    try:
        logger.info(f"Loading data from database with query: {query}")
        df = load_data_from_database(_connection_string, query)
        
        # Standardize column names
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        with st.spinner("Preprocessing data..."):
            df = DataProcessor.preprocess_data(df)
            
        return df
        
    except Exception as e:
        st.error(f"Error loading database data: {str(e)}. Please check the connection string and query.")
        logger.exception("Database loading error")
        return None

@st.cache_data
def load_and_preprocess_api(_api_config: dict) -> Optional[pd.DataFrame]:
    """
    Load and preprocess data from an API.
    Args:
        _api_config (dict): API configuration (hashable for caching).
    Returns:
        Optional[pd.DataFrame]: The preprocessed DataFrame or None if loading fails.
    """
    try:
        logger.info(f"Loading data from API with config: {_api_config}")
        df = load_data_from_api(_api_config)
        
        # Standardize column names
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        with st.spinner("Preprocessing data..."):
            df = DataProcessor.preprocess_data(df)
            
        return df
        
    except Exception as e:
        st.error(f"Error loading API data: {str(e)}. Please verify the API configuration.")
        logger.exception("API loading error")
        return None

def load_data() -> Optional[pd.DataFrame]:
    """
    Load and preprocess data based on the selected data source.
    Returns:
        Optional[pd.DataFrame]: The preprocessed DataFrame or None if loading fails.
    """
    logger.info("Starting load_data function")
    
    if not hasattr(st.session_state, "state") or not st.session_state.state:
        st.error("Session state not initialized. Please restart the app.")
        logger.error("Session state not initialized")
        return None
        
    session_state = st.session_state.state
    
    if not hasattr(session_state, "data_source"):
        st.error("No data source selected. Please select a data source in the control panel.")
        logger.error("No data source selected")
        return None
        
    data_source = session_state.data_source
    logger.info(f"Using data source: {data_source}")
    
    # Validate inputs
    is_valid, error_message = validate_inputs(data_source, session_state)
    if not is_valid:
        st.error(error_message)
        logger.error(f"Input validation failed: {error_message}")
        return None
        
    # Load and preprocess data
    with st.spinner("Loading data..."):
        try:
            if data_source == "csv":
                data = load_and_preprocess_csv(session_state.uploaded_file)
            elif data_source == "database":
                query = getattr(session_state, "query", "SELECT * FROM demand_data")
                data = load_and_preprocess_database(session_state.connection_string, query)
            elif data_source == "api":
                data = load_and_preprocess_api(session_state.api_config)
            else:
                st.error(f"Unsupported data source: {data_source}")
                logger.error(f"Unsupported data source: {data_source}")
                return None
                
            if data is not None:
                # Update session state
                session_state.data = data
                logger.info(f"Data loaded successfully with {len(data)} rows")
                return data
            else:
                logger.error("Data loading failed")
                return None
                
        except Exception as e:
            st.error(f"Unexpected error loading data: {str(e)}")
            logger.exception("Unexpected error in load_data")
            return None