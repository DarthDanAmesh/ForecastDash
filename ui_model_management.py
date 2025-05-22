# ui_model_management.py
import streamlit as st
from typing import Optional, Dict, Any
from cls_session_management import SessionState
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model_file(file: Any) -> tuple[bool, str]:
    """Validate uploaded model file."""
    if file is None:
        return False, "No file uploaded. Please select a .pt file."
    if not file.name.endswith(".pt"):
        return False, "Invalid file format. Please upload a .pt file."
    if file.size > 50 * 1024 * 1024:
        return False, "File too large. Please upload a file smaller than 50 MB."
    return True, ""

def save_model(model: Any, model_type: str) -> Optional[bytes]:
    """Save model to in-memory buffer."""
    try:
        buffer = torch.save(model, f"{model_type}.pt")
        with open(f"{model_type}.pt", 'rb') as f:
            return f.read()
    except Exception as e:
        st.error(f"Failed to save model '{model_type}': {str(e)}.", icon="🚨")
        logger.error(f"Model save error: {str(e)}")
        return None

def load_model(file: Any, model_type: str) -> Optional[Any]:
    """Load model from uploaded file."""
    try:
        with open("temp.pt", "wb") as f:
            f.write(file.read())
        model = torch.load("temp.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model '{model_type}': {str(e)}.", icon="🚨")
        logger.error(f"Model load error: {str(e)}")
        return None

def render_model_selection() -> Optional[str]:
    """Render model selection UI."""
    models = st.session_state.state.models
    if not models:
        st.warning("No models available. Generate a forecast to create a model.", icon="⚠️")
        return None
    return st.selectbox(
        "Select Model",
        list(models.keys()),
        help="Choose a trained model to load or save."
    )

def show_model_management() -> None:
    """Render model management UI for Technical Mode."""
    if 'state' not in st.session_state or not isinstance(st.session_state.state, SessionState):
        st.error("Session state not initialized. Please restart the app.", icon="🚨")
        return
    if st.session_state.get('mode', 'Simple') != "Technical":
        return
    
    st.header("Model Management", help="Load or save forecasting models.")
    
    model_type = render_model_selection()
    
    with st.expander("Load Model", expanded=False):
        st.subheader("Upload Model File")
        uploaded_model = st.file_uploader(
            "Choose a .pt model file",
            type=['pt'],
            help="Upload a .pt file containing a trained DeepAR or XGBoost model."
        )
        if uploaded_model:
            with st.spinner("Loading model..."):
                is_valid, message = validate_model_file(uploaded_model)
                if not is_valid:
                    st.error(message, icon="🚨")
                    return
                loaded_model = load_model(uploaded_model, model_type or "DeepAR")
                if loaded_model:
                    st.session_state.state.models[model_type or "DeepAR"] = {'model': loaded_model, 'trained_at': pd.Timestamp.now()}
                    st.success(f"Model '{model_type or 'DeepAR'}' loaded successfully!", icon="✅")
    
    with st.expander("Save Model", expanded=False):
        st.subheader("Save Model")
        if model_type:
            if st.button("Export Model"):
                with st.spinner("Saving model..."):
                    model = st.session_state.state.models.get(model_type, {}).get('model')
                    if not model:
                        st.error(f"No model found for '{model_type}'.", icon="🚨")
                        return
                    model_bytes = save_model(model, model_type)
                    if model_bytes:
                        st.download_button(
                            label="Download .pt File",
                            data=model_bytes,
                            file_name=f"{model_type}_model.pt",
                            mime="application/octet-stream"
                        )
                        st.success(f"Model '{model_type}' exported successfully!", icon="✅")
        else:
            st.info("Select a model to save.", icon="ℹ️")