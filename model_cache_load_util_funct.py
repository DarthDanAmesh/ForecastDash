import pandas as pd
import streamlit as st
import pickle
from pathlib import Path
import hashlib
import json
from typing import Optional, Any

def model_cache_path(model_name: str) -> Path:
    """Generate standardized cache file path"""
    cache_dir = Path("saved_models")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{model_name.lower()}_cache.pkl"

def get_data_fingerprint(df: pd.DataFrame, config: dict) -> str:
    """Create unique hash of data and configuration"""
    data_str = df.to_json() + json.dumps(config, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

def save_model_cache(model: Any, fingerprint: str, model_name: str) -> None:
    """Save model with validation fingerprint"""
    cache_data = {
        "model": model,
        "fingerprint": fingerprint,
        "timestamp": pd.Timestamp.now()
    }
    with open(model_cache_path(model_name), "wb") as f:
        pickle.dump(cache_data, f)

def load_model_cache(fingerprint: str, model_name: str) -> Optional[Any]:
    """Load model only if fingerprint matches"""
    cache_file = model_cache_path(model_name)
    if not cache_file.exists():
        return None
    
    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)
    
    if cache_data.get("fingerprint") == fingerprint:
        return cache_data["model"]
    return None