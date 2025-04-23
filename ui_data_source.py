import streamlit as st

# UI Components
def show_data_source_selection():
    st.sidebar.header("Data Source Configuration")
    data_source = st.sidebar.radio("Select Data Source", 
                                  ["CSV Upload", "Database", "API"])
    
    if data_source == "CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
        if uploaded_file:
            st.session_state.state.uploaded_file = uploaded_file
            st.session_state.state.data_source = "csv"
    
    elif data_source == "Database":
        db_type = st.sidebar.selectbox("Database Type", ["SQLite", "PostgreSQL", "MySQL"])
        connection_string = st.sidebar.text_input("Connection String", 
                                                "sqlite:///demand_data.sqlite")
        query = st.sidebar.text_area("SQL Query (optional)", 
                                   "SELECT * FROM demand_data")
        
        if st.sidebar.button("Connect to Database"):
            st.session_state.state.connection_string = connection_string
            st.session_state.state.data_source = "database"
            st.session_state.state.query = query
    
    elif data_source == "API":
        api_endpoint = st.sidebar.text_input("API Endpoint")
        api_key = st.sidebar.text_input("API Key (if required)", type="password")
        
        if st.sidebar.button("Connect to API"):
            st.session_state.state.api_config = {
                "endpoint": api_endpoint,
                "key": api_key
            }
            st.session_state.state.data_source = "api"