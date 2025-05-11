import streamlit as st
import pickle
import io
from typing import Optional, Dict, Any
from cls_session_management import SessionState

def validate_model_file(file: Any) -> tuple[bool, str]:
    """
    Validate the uploaded model file.

    Args:
        file: Uploaded file object.

    Returns:
        tuple[bool, str]: (is_valid, message) indicating if the file is valid and any error message.
    """
    if file is None:
        return False, "No file uploaded. Please select a .pkl file."
    if not file.name.endswith(".pkl"):
        return False, "Invalid file format. Please upload a .pkl file."
    if file.size > 50 * 1024 * 1024:  # 50 MB limit
        return False, "File too large. Please upload a file smaller than 50 MB."
    return True, ""

def save_model(model: Any, model_type: str) -> Optional[bytes]:
    """
    Save a model to an in-memory buffer and return the serialized bytes.

    Args:
        model: The model object to save.
        model_type (str): The model type (used for file naming).

    Returns:
        Optional[bytes]: Serialized model bytes or None if saving fails.
    """
    try:
        with io.BytesIO() as buffer:
            pickle.dump(model, buffer)
            return buffer.getvalue()
    except Exception as e:
        st.error(f"Failed to serialize model '{model_type}': {str(e)}. Please ensure the model is picklable.")
        return None

def load_model(file: Any, model_type: str) -> Optional[Any]:
    """
    Load a model from an uploaded file.

    Args:
        file: Uploaded file object.
        model_type (str): The model type to assign the loaded model.

    Returns:
        Optional[Any]: Loaded model object or None if loading fails.
    """
    try:
        loaded_model = pickle.load(file)
        return loaded_model
    except pickle.UnpicklingError:
        st.error(f"Invalid .pkl file for '{model_type}'. Please upload a valid pickled model.")
        return None
    except Exception as e:
        st.error(f"Error loading model '{model_type}': {str(e)}. Please verify the file.")
        return None

def render_model_selection() -> Optional[str]:
    """
    Render the model selection UI and return the selected model type.

    Returns:
        Optional[str]: Selected model type or None if no models are available.
    """
    models = st.session_state.state.models
    if not models:
        st.warning("⚠️ No models available. Please train a model in the 'Model Training' section or upload a model below.")
        return None
    return st.selectbox(
        "📂 Select Model",
        list(models.keys()),
        help="Choose a trained model to load or save."
    )

def show_model_management() -> None:
    """
    Render the UI for model management, including loading and saving models.

    Only shown in Technical Mode; hidden in Simple Mode.
    """
    # Check session state and mode
    if 'state' not in st.session_state or not isinstance(st.session_state.state, SessionState):
        st.error("Session state not initialized. Please restart the app.")
        return
    if st.session_state.get('mode', 'Simple') != "Technical":
        return  # Hide UI in Simple Mode

    st.header("Model Management", help="Load or save trained models for demand forecasting.")

    # Model selection
    model_type = render_model_selection()

    # Model loading and saving sections
    with st.expander("📥 Load Model", expanded=False):
        st.subheader("Upload a Model File", help="Upload a .pkl file containing a trained model.")
        uploaded_model = st.file_uploader(
            "Choose a .pkl model file",
            type=['pkl'],
            help="Select a .pkl file containing a trained model (e.g., scikit-learn, Prophet)."
        )
        if uploaded_model:
            with st.spinner("Loading model..."):
                is_valid, message = validate_model_file(uploaded_model)
                if not is_valid:
                    st.error(message)
                    st.toast(f"❌ {message}")
                    return
                loaded_model = load_model(uploaded_model, model_type or "uploaded_model")
                if loaded_model:
                    try:
                        st.session_state.state.models[model_type or "uploaded_model"] = {'model': loaded_model}
                        st.success(f"Model '{model_type or 'uploaded_model'}' loaded successfully!")
                        st.toast("✅ Model loaded successfully!")
                    except AttributeError:
                        st.error("Session state error. Please restart the app.")

    with st.expander("💾 Save Model", expanded=False):
        st.subheader("Save Selected Model", help="Export a trained model as a .pkl file for reuse.")
        if model_type:
            if st.button("⬇️ Export Model", help="Download the selected model as a .pkl file."):
                with st.spinner("Saving model..."):
                    model = st.session_state.state.models.get(model_type, {}).get('model')
                    if not model:
                        st.error(f"No model found for '{model_type}'. Please select a valid model.")
                        st.toast("❌ No model found.")
                        return
                    model_bytes = save_model(model, model_type)
                    if model_bytes:
                        st.download_button(
                            label="⬇️ Download .pkl File",
                            data=model_bytes,
                            file_name=f"{model_type}_model.pkl",
                            mime="application/octet-stream",
                            help="Download the model as a .pkl file."
                        )
                        st.success(f"Model '{model_type}' exported successfully!")
                        st.toast("✅ Model exported successfully!")
        else:
            st.info("Select a model above to enable saving.")