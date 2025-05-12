import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from threading import Lock
import hashlib
import json

class SessionState:
    """Manages application state for the demand planning toolkit, ensuring thread-safety and validation."""

    _VALID_DATA_SOURCES = {'csv', 'database', 'api'}
    _lock = Lock()  # Thread-safe lock for state modifications

    def __init__(self) -> None:
        """Initialize session state with default values."""
        self._data: Optional[pd.DataFrame] = None
        self._models: Dict[str, Any] = {}
        self._forecasts: Dict[str, pd.DataFrame] = {}
        self._data_source: Optional[str] = None
        self._connection_string: Optional[str] = None
        self._uploaded_file: Optional[Any] = None
        self._model_cache: Dict[str, Any] = {}
        self._api_config: Optional[Dict[str, Any]] = None
        self._data_fingerprints: Dict[str, str] = {}
        self._accuracy_data: Dict[str, Dict[str, float]] = {}

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the preprocessed DataFrame."""
        return self._data

    @data.setter
    def data(self, value: Optional[pd.DataFrame]) -> None:
        """Set the preprocessed DataFrame with validation."""
        with self._lock:
            if value is not None and not isinstance(value, pd.DataFrame):
                st.error("Invalid data: Must be a Pandas DataFrame.")
                return
            self._data = value
            if value is not None:
                self._update_data_fingerprint()

    @property
    def data_source(self) -> Optional[str]:
        """Get the data source."""
        return self._data_source

    @data_source.setter
    def data_source(self, value: Optional[str]) -> None:
        """Set the data source with validation."""
        with self._lock:
            if value is not None and value not in self._VALID_DATA_SOURCES:
                st.error(f"Invalid data source: Must be one of {', '.join(self._VALID_DATA_SOURCES)}.")
                return
            self._data_source = value

    @property
    def connection_string(self) -> Optional[str]:
        """Get the database connection string."""
        return self._connection_string

    @connection_string.setter
    def connection_string(self, value: Optional[str]) -> None:
        """Set the database connection string with validation."""
        with self._lock:
            if value is not None and not isinstance(value, str):
                st.error("Invalid connection string: Must be a string.")
                return
            self._connection_string = value

    @property
    def uploaded_file(self) -> Optional[Any]:
        """Get the uploaded file object."""
        return self._uploaded_file

    @uploaded_file.setter
    def uploaded_file(self, value: Optional[Any]) -> None:
        """Set the uploaded file object."""
        with self._lock:
            self._uploaded_file = value

    @property
    def api_config(self) -> Optional[Dict[str, Any]]:
        """Get the API configuration."""
        return self._api_config

    @api_config.setter
    def api_config(self, value: Optional[Dict[str, Any]]) -> None:
        """Set the API configuration with validation."""
        with self._lock:
            if value is not None:
                if not isinstance(value, dict) or 'url' not in value:
                    st.error("Invalid API configuration: Must be a dictionary with a 'url' key.")
                    return
            self._api_config = value


    @property
    def models(self) -> Dict[str, Any]:
        """Get the dictionary of trained models."""
        return self._models

    @models.setter
    def models(self, value: Dict[str, Any]) -> None:
        with self._lock:
            self._models = value

    @property
    def forecasts(self) -> Dict[str, pd.DataFrame]:
        """Get the dictionary of forecast results."""
        return self._forecasts

    @property
    def model_cache(self) -> Dict[str, Any]:
        """Get the model cache."""
        return self._model_cache

    @property
    def data_fingerprints(self) -> Dict[str, str]:
        """Get the data fingerprints."""
        return self._data_fingerprints

    @property
    def accuracy_data(self) -> Dict[str, Dict[str, float]]:
        """Get the accuracy data."""
        return self._accuracy_data

    def _update_data_fingerprint(self) -> None:
        """
        Update the data fingerprint based on the current DataFrame.

        Uses MD5 hash of serialized DataFrame to track data changes.
        """
        if self._data is not None:
            try:
                # Serialize DataFrame to string for hashing
                data_str = self._data.to_json()
                fingerprint = hashlib.md5(data_str.encode()).hexdigest()
                self._data_fingerprints['current'] = fingerprint
            except Exception as e:
                st.warning(f"Error updating data fingerprint: {str(e)}.")

    def reset(self, full_reset: bool = True) -> None:
        """
        Reset session state to default values.

        Args:
            full_reset (bool): If True, reset all attributes; if False, reset only data-related attributes.
        """
        with self._lock:
            self._data = None
            self._models = {}
            self._forecasts = {}
            self._model_cache = {}
            self._data_fingerprints = {}
            self._accuracy_data = {}
            if full_reset:
                self._data_source = None
                self._connection_string = None
                self._uploaded_file = None
                self._api_config = None
            st.success("Session state reset successfully.")

    def initialize_session(self) -> None:
        """
        Ensure session state is properly initialized in Streamlit.

        Checks if st.session_state.state exists and initializes it if needed.
        """
        with self._lock:
            if 'state' not in st.session_state:
                st.session_state.state = self
                st.success("Session state initialized successfully.")
            elif not isinstance(st.session_state.state, SessionState):
                st.error("Session state corrupted. Reinitializing...")
                st.session_state.state = self

    def validate_state(self) -> tuple[bool, str]:
        """
        Validate the current session state.

        Returns:
            tuple[bool, str]: (is_valid, message) indicating if the state is valid and any error message.
        """
        with self._lock:
            if self._data_source is not None and self._data_source not in self._VALID_DATA_SOURCES:
                return False, f"Invalid data source: {self._data_source}"
            if self._data is not None and not isinstance(self._data, pd.DataFrame):
                return False, "Invalid data: Must be a Pandas DataFrame."
            if self._api_config is not None and ('url' not in self._api_config):
                return False, "Invalid API configuration: Missing 'url' key."
            if self._connection_string is not None and not isinstance(self._connection_string, str):
                return False, "Invalid connection string: Must be a string."
            return True, ""

    @classmethod
    def get_or_create(cls):
        """Get the current session state or create a new one if it doesn't exist."""
        if 'state' not in st.session_state or not isinstance(st.session_state.state, cls):
            st.session_state.state = cls()
        return st.session_state.state