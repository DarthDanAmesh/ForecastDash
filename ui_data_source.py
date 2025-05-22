# ui_data_source.py
import streamlit as st
from typing import Optional
from urllib.parse import urlparse
from cls_session_management import SessionState

def validate_connection_string(db_type: str, connection_string: str) -> tuple[bool, str]:
    """Validate the database connection string format."""
    if not connection_string:
        return False, "Connection string cannot be empty."
    if db_type == "SQLite" and not connection_string.startswith("sqlite:///"):
        return False, "SQLite connection string must start with 'sqlite:///'."
    if db_type in ["PostgreSQL", "MySQL"] and "://" not in connection_string:
        return False, f"{db_type} connection string must include a scheme (e.g., 'postgresql://')."
    return True, ""

def validate_api_config(endpoint: str, key: Optional[str] = None) -> tuple[bool, str]:
    """Validate the API configuration."""
    if not endpoint:
        return False, "API endpoint cannot be empty."
    try:
        result = urlparse(endpoint)
        if not all([result.scheme, result.netloc]):
            return False, "Invalid API endpoint. Must be a valid URL (e.g., 'https://api.example.com/data')."
    except ValueError:
        return False, "Invalid API endpoint format."
    return True, ""

def render_csv_ui() -> None:
    """Render the UI for CSV file upload."""
    st.markdown("#### 📁 Upload a CSV File", help="Upload a CSV file containing demand data (e.g., columns: date, demand).")
    uploaded_file = st.file_uploader(
        "Select a CSV file",
        type=["csv"],
        help="Supported format: CSV. Ensure the file includes required columns (e.g., date, demand)."
    )
    if uploaded_file:
        try:
            st.session_state.state.uploaded_file = uploaded_file
            st.session_state.state.data_source = "csv"
            st.success("CSV file uploaded successfully! Click 'Load Data' to process.", icon="✅")
            st.toast("✅ CSV file ready for loading!")
        except AttributeError:
            st.error("Session state not initialized. Please restart the app.", icon="🚨")
            st.session_state.state.uploaded_file = None

def render_database_ui() -> None:
    """Render the UI for database connection."""
    st.markdown("#### 🗄️ Connect to a Database", help="Enter details to connect to a database containing demand data.")
    db_type = st.selectbox(
        "Database Type",
        ["SQLite", "PostgreSQL", "MySQL"],
        help="Select the database type you are connecting to."
    )
    connection_string = st.text_input(
        "Connection String",
        placeholder="e.g., sqlite:///demand_data.sqlite or postgresql://user:password@host:port/db",
        help="Enter the connection string for your database."
    )
    query = st.text_area(
        "SQL Query (Optional)",
        placeholder="SELECT * FROM demand_data",
        help="Enter a SQL query to fetch data. Leave blank to use the default query."
    )

    if st.button("🔌 Connect to Database", help="Validate and save database connection details."):
        is_valid, message = validate_connection_string(db_type, connection_string)
        if not is_valid:
            st.error(message, icon="🚨")
            st.toast(f"❌ {message}")
            return
        try:
            st.session_state.state.connection_string = connection_string
            st.session_state.state.data_source = "database"
            st.session_state.state.query = query if query.strip() else "SELECT * FROM demand_data"
            st.success(f"Connected to {db_type} database! Click 'Load Data' to fetch data.", icon="✅")
            st.toast(f"✅ {db_type} database connection configured!")
        except AttributeError:
            st.error("Session state not initialized. Please restart the app.", icon="🚨")
            st.session_state.state.connection_string = None

def render_api_ui() -> None:
    """Render the UI for API connection."""
    st.markdown("#### 🌐 Connect to an API", help="Enter details to fetch demand data from an API.")
    api_endpoint = st.text_input(
        "API Endpoint",
        placeholder="https://api.example.com/data",
        help="Enter the API endpoint URL."
    )
    api_key = st.text_input(
        "API Key (Optional)",
        type="password",
        help="Enter the API key if required."
    )

    if st.button("🔌 Connect to API", help="Validate and save API connection details."):
        is_valid, message = validate_api_config(api_endpoint, api_key)
        if not is_valid:
            st.error(message, icon="🚨")
            st.toast(f"❌ {message}")
            return
        try:
            st.session_state.state.api_config = {
                "url": api_endpoint,
                "headers": {"Authorization": f"Bearer {api_key}"} if api_key else {}
            }
            st.session_state.state.data_source = "api"
            st.success("API connection configured! Click 'Load Data' to fetch data.", icon="✅")
            st.toast("✅ API connection ready!")
        except AttributeError:
            st.error("Session state not initialized. Please restart the app.", icon="🚨")
            st.session_state.state.api_config = None

def show_data_source_selection() -> None:
    """Render the sidebar UI for selecting and configuring a data source."""
    with st.sidebar:
        st.markdown("### Configure Data Source", help="Select a method to load your demand data.")
        
        if 'state' not in st.session_state or not isinstance(st.session_state.state, SessionState):
            st.error("Session state not initialized. Please restart the app.", icon="🚨")
            return

        mode = st.session_state.get('mode', 'Simple')
        if mode == "Simple":
            data_source = "📁 CSV Upload"
            st.markdown("**Simple Mode: CSV Upload**", help="Upload a CSV file for quick analysis.")
        else:
            data_source = st.radio(
                "Choose how to load data:",
                ["📁 CSV Upload", "🗄️ Database", "🌐 API"],
                index=0,
                help="Select a data source: CSV, database, or API endpoint."
            )

        st.markdown("---")

        if data_source == "📁 CSV Upload":
            render_csv_ui()
        elif data_source == "🗄️ Database":
            render_database_ui()
        elif data_source == "🌐 API":
            render_api_ui()