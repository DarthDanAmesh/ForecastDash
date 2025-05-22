import streamlit as st
import pandas as pd
from typing import Optional, Dict
from sqlalchemy import create_engine
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_file(file_path: str) -> tuple[bool, str]:
    """Validate the CSV file path."""
    if not file_path:
        return False, "No file path provided. Please upload a CSV file."
    if not file_path.endswith(".csv"):
        return False, "Invalid file format. Please upload a CSV file."
    return True, ""

@st.cache_data(show_spinner=False)
def load_data_from_csv(_file_path: str) -> Optional[pd.DataFrame]:
    """Load data from a CSV file path."""
    is_valid, message = validate_file(_file_path)
    if not is_valid:
        st.error(message, icon="🚨")
        logger.error(message)
        return None

    with st.spinner("Loading CSV file..."):
        try:
            df = pd.read_csv(_file_path)
            
            if df.empty:
                st.warning("Loaded file is empty. Please upload a file with data.", icon="⚠️")
                logger.warning("Empty CSV file loaded")
                return None
            
            logger.info("CSV file loaded successfully")
            return df
        except pd.errors.ParserError:
            st.error("Invalid CSV format. Ensure the file is a valid CSV.", icon="🚨")
            logger.error("CSV parsing error")
            return None
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}.", icon="🚨")
            logger.error(f"CSV loading error: {str(e)}")
            return None

@st.cache_data(show_spinner=False)
def load_data_from_database(_connection_string: str, query: str = "SELECT * FROM demand_data") -> Optional[pd.DataFrame]:
    """Load data from a database using a connection string and SQL query."""
    if not _connection_string:
        st.error("No connection string provided. Enter a valid database connection string.", icon="🚨")
        logger.error("Missing database connection string")
        return None

    with st.spinner("Loading database data..."):
        try:
            engine = create_engine(_connection_string)
            df = pd.read_sql(query, engine)
            
            if df.empty:
                st.warning("No data returned from the query. Check the query or database.", icon="⚠️")
                logger.warning("Empty database query result")
                return None
            
            logger.info("Database data loaded successfully")
            return df
        except Exception as e:
            st.error(f"Error loading database data: {str(e)}.", icon="🚨")
            logger.error(f"Database loading error: {str(e)}")
            return None

@st.cache_data(show_spinner=False)
def load_data_from_api(_api_config: Dict) -> Optional[pd.DataFrame]:
    """Load data from an API using the provided configuration."""
    if not _api_config or 'url' not in _api_config:
        st.error("Invalid API configuration. Provide a configuration with a 'url' key.", icon="🚨")
        logger.error("Invalid API configuration")
        return None

    with st.spinner("Loading API data..."):
        try:
            response = requests.get(
                _api_config['url'],
                headers=_api_config.get('headers', {}),
                params=_api_config.get('params', {})
            )
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            if df.empty:
                st.warning("No data returned from the API. Check the configuration or endpoint.", icon="⚠️")
                logger.warning("Empty API response")
                return None
            
            logger.info("API data loaded successfully")
            return df
        except Exception as e:
            st.error(f"Error loading API data: {str(e)}.", icon="🚨")
            logger.error(f"API loading error: {str(e)}")
            return None