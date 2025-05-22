# simple_mode_utils.py
import pandas as pd
from constants import STANDARD_COLUMNS
import streamlit as st

def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter data based on sidebar selections."""
    if data is None or 'filters' not in st.session_state:
        return data
    
    filtered_data = data.copy()
    filters = st.session_state.filters
    
    if filters['materials']:
        filtered_data = filtered_data[filtered_data[STANDARD_COLUMNS['material']].isin(filters['materials'])]
    
    if filters['countries']:
        filtered_data = filtered_data[filtered_data[STANDARD_COLUMNS['country']].isin(filters['countries'])]
    
    # Use st.session_state.date_range directly
    if 'date_range' in st.session_state and st.session_state.date_range:
        date_range = st.session_state.date_range
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = filtered_data[
                (pd.to_datetime(filtered_data[STANDARD_COLUMNS['date']]) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(filtered_data[STANDARD_COLUMNS['date']]) <= pd.to_datetime(end_date))
            ]
    
    return filtered_data

def calculate_performance_gaps(ts: pd.Series) -> pd.Series:
    """Calculate performance gaps as percentage deviation from rolling mean."""
    rolling_mean = ts.rolling(window=3, min_periods=1).mean()
    gaps = ((ts - rolling_mean) / rolling_mean * 100).fillna(0)
    return gaps