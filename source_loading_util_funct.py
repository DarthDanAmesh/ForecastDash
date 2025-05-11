import streamlit as st
import pandas as pd
from typing import Optional, Dict
from sqlalchemy import create_engine
import requests
from io import StringIO

def validate_file(file) -> tuple[bool, str]:
    """
    Validate the uploaded file.

    Args:
        file: Uploaded file object.

    Returns:
        tuple[bool, str]: (is_valid, message) indicating if the file is valid and any error message.
    """
    if file is None:
        return False, "No file provided. Please upload a CSV or Excel file."
    if not file.name.endswith((".csv", ".xlsx")):
        return False, "Invalid file format. Please upload a CSV or Excel file."
    if file.size > 100 * 1024 * 1024:  # 100 MB limit
        return False, "File too large. Please upload a file smaller than 100 MB."
    return True, ""

@st.cache_data
def load_data_from_csv(_file) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV or Excel file.

    Args:
        _file: Uploaded file object (hashable for caching).

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if loading fails.
    """
    # Validate file
    is_valid, message = validate_file(_file)
    if not is_valid:
        st.error(message)
        return None

    with st.spinner("Loading CSV/Excel file..."):
        try:
            if _file.name.endswith(".csv"):
                df = pd.read_csv(_file)
            else:
                df = pd.read_excel(_file)
            
            if df.empty:
                st.warning("Loaded file is empty. Please upload a file with data.")
                return None
            
            st.success("CSV/Excel file loaded successfully!")
            return df
        except pd.errors.ParserError:
            st.error("Invalid file format. Please ensure the file is a valid CSV or Excel file.")
            return None
        except Exception as e:
            st.error(f"Error loading file: {str(e)}. Please verify the file content.")
            return None

@st.cache_data
def load_data_from_database(_connection_string: str, query: str = "SELECT * FROM demand_data") -> Optional[pd.DataFrame]:
    """
    Load data from a database using a connection string and SQL query.

    Args:
        _connection_string (str): Database connection string (hashable for caching).
        query (str): SQL query to execute (default: 'SELECT * FROM demand_data').

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if loading fails.
    """
    if not _connection_string:
        st.error("No connection string provided. Please enter a valid database connection string.")
        return None

    with st.spinner("Loading database data..."):
        try:
            engine = create_engine(_connection_string)
            df = pd.read_sql(query, engine)
            
            if df.empty:
                st.warning("No data returned from the query. Please check the query or database.")
                return None
            
            st.success("Database data loaded successfully!")
            return df
        except sqlalchemy.exc.OperationalError:
            st.error("Database connection failed. Please verify the connection string and ensure the database is accessible.")
            return None
        except sqlalchemy.exc.ProgrammingError:
            st.error("Invalid SQL query. Please check the query syntax and table/column names.")
            return None
        except Exception as e:
            st.error(f"Error loading database data: {str(e)}. Please verify the connection and query.")
            return None

def load_data_from_api(api_config: Dict) -> Optional[pd.DataFrame]:
    """
    Load data from an API using the provided configuration.

    Args:
        api_config (Dict): Configuration dictionary with keys 'url', 'headers' (optional), 'params' (optional).

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if loading fails.
    """
    if not api_config or 'url' not in api_config:
        st.error("Invalid API configuration. Please provide a configuration with a 'url' key.")
        return None

    with st.spinner("Loading API data..."):
        try:
            response = requests.get(
                api_config['url'],
                headers=api_config.get('headers', {}),
                params=api_config.get('params', {})
            )
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Assume JSON response with tabular data
            data = response.json()
            
            # Convert to DataFrame (adjust based on actual API response format)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            if df.empty:
                st.warning("No data returned from the API. Please check the API configuration or endpoint.")
                return None
            
            st.success("API data loaded successfully!")
            return df
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}. Please verify the URL, headers, and network connection.")
            return None
        except ValueError:
            st.error("Invalid API response format. Please ensure the API returns valid JSON data.")
            return None
        except Exception as e:
            st.error(f"Error loading API data: {str(e)}. Please verify the API configuration.")
            return None