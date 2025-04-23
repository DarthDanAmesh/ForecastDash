import streamlit as st

def show_data_source_selection():
    with st.sidebar:
        st.markdown("### 🗂️ Configure Data Source")

        data_source = st.radio("Choose how to load data:", 
                               ["📁 CSV Upload", "🗄️ Database", "🌐 API"],
                               index=0)

        st.markdown("---")

        if data_source == "📁 CSV Upload":
            st.markdown("#### Upload a CSV File")
            uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])
            if uploaded_file:
                st.session_state.state.uploaded_file = uploaded_file
                st.session_state.state.data_source = "csv"
                st.toast("CSV file uploaded successfully!")

        elif data_source == "🗄️ Database":
            st.markdown("#### Connect to a Database")
            db_type = st.selectbox("Database Type", ["SQLite", "PostgreSQL", "MySQL"])
            connection_string = st.text_input(
                "Connection String",
                placeholder="e.g., sqlite:///demand_data.sqlite"
            )
            query = st.text_area(
                "SQL Query (Optional)",
                placeholder="SELECT * FROM demand_data"
            )

            if st.button("🔌 Connect to Database"):
                st.session_state.state.connection_string = connection_string
                st.session_state.state.data_source = "database"
                st.session_state.state.query = query
                st.toast(f"Connected to {db_type} database!")

        elif data_source == "🌐 API":
            st.markdown("#### Connect to an API")
            api_endpoint = st.text_input("API Endpoint", placeholder="https://api.example.com/data")
            api_key = st.text_input("API Key (if needed)", type="password")

            if st.button("🔌 Connect to API"):
                st.session_state.state.api_config = {
                    "endpoint": api_endpoint,
                    "key": api_key
                }
                st.session_state.state.data_source = "api"
                st.toast("API connection configured!")
