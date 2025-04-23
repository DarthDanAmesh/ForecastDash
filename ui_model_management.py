import streamlit as st
import tempfile
import pickle
import os

def show_model_management():
    st.header("Model Management")

    if not st.session_state.state.models:
        st.warning("⚠️ No models available. Train or upload a model to get started.")
        return

    model_type = st.selectbox("📂 Select Model", list(st.session_state.state.models.keys()))

    tab1, tab2 = st.tabs(["📥 Load Model", "💾 Save Model"])

    with tab2:
        st.subheader("💾 Save Selected Model")
        if st.button("Export Model"):
            try:
                model = st.session_state.state.models[model_type]['model']
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                    pickle.dump(model, tmp)
                    tmp_path = tmp.name
                with open(tmp_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download .pkl File",
                        data=f,
                        file_name=f"{model_type}_model.pkl",
                        mime="application/octet-stream"
                    )
                os.unlink(tmp_path)
                st.toast("✅ Model exported successfully!")
            except Exception as e:
                st.error(f"❌ Failed to save model: {str(e)}")

    with tab1:
        st.subheader("📥 Load Model File")
        uploaded_model = st.file_uploader("Choose a .pkl model file", type=['pkl'])
        if uploaded_model:
            try:
                loaded_model = pickle.load(uploaded_model)
                if model_type in st.session_state.state.models:
                    st.session_state.state.models[model_type]['model'] = loaded_model
                else:
                    st.session_state.state.models[model_type] = {'model': loaded_model}
                st.toast(f"✅ Model '{model_type}' loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
