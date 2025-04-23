import streamlit as st
import tempfile
import pickle
import os

def show_model_management():
    st.header("Model Management")
    
    if not st.session_state.state.models:
        st.warning("No models available for management.")
        return
    
    model_type = st.selectbox("Select Model", list(st.session_state.state.models.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Model"):
            model_info = st.session_state.state.models[model_type]
            model = model_info['model']
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                pickle.dump(model, tmp)
                tmp_path = tmp.name
            
            with open(tmp_path, "rb") as f:
                st.download_button(
                    label="Download Model",
                    data=f,
                    file_name=f"{model_type}_demand_model.pkl",
                    mime="application/octet-stream"
                )
            
            os.unlink(tmp_path)
            st.success("Model saved successfully!")
    
    with col2:
        uploaded_model = st.file_uploader("Upload Model", type=['pkl'])
        if uploaded_model and model_type:
            try:
                model = pickle.load(uploaded_model)
                st.session_state.state.models[model_type]['model'] = model
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")