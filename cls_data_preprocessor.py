# cls_data_preprocessor.py
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict
from column_config import STANDARD_COLUMNS, standardize_column_names, COLUMN_ALIASES

class DataProcessor:
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: List[str], optional_cols: List[str] = None) -> tuple[bool, str]:
        """
        Validate the presence of required and optional columns in the DataFrame.
        """
        if df is None:
            return False, "Input DataFrame is None."
        
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            return False, f"Missing required columns: {', '.join(missing_required)}."
            
        if optional_cols:
            missing_optional = [col for col in optional_cols if col not in df.columns]
            if missing_optional:
                return True, f"Warning: Missing optional columns: {', '.join(missing_optional)}."
                
        return True, ""

    @staticmethod
    def convert_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        """
        Convert specified columns to datetime format if not already datetime.
        """
        for col in [c for c in date_cols if c in df.columns]:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].isna().all():
                        st.warning(f"Column '{col}' could not be converted to datetime. All values are NaT.")
                except Exception as e:
                    st.warning(f"Error converting column '{col}' to datetime: {str(e)}. Values set to NaT.")
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, fill_rules: Dict[str, str]) -> pd.DataFrame:
        """
        Handle missing values in specified columns according to fill rules.
        """
        for col, fill_value in fill_rules.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
        return df

    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features such as Delivery_Delay.
        """
        planned_date_col = STANDARD_COLUMNS['planned_delivery_date']
        actual_date_col = STANDARD_COLUMNS['delivery_date']
        
        if planned_date_col in df.columns and actual_date_col in df.columns:
            try:
                df['Delivery_Delay'] = (df[actual_date_col] - df[planned_date_col]).dt.days
            except Exception as e:
                st.warning(f"Error calculating Delivery_Delay: {str(e)}. Column not added.")
        return df

    @staticmethod
    def preprocess_data(df: pd.DataFrame, 
                       required_cols: List[str] = [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand']],
                       date_cols: List[str] = [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['planned_delivery_date'], 
                                              STANDARD_COLUMNS['delivery_date'], STANDARD_COLUMNS['order_date']],
                       fill_rules: Dict[str, str] = {
                           'Customer Reference': 'Customer Reference', 
                           'Customer Material': 'Customer Material',
                           STANDARD_COLUMNS['country']: 'Unknown'
                       }) -> Optional[pd.DataFrame]:
        """
        Clean and preprocess the input data.
        """
        if df is None:
            st.error("Input DataFrame is None.")
            return None
            
        # Standardize column names
        df = standardize_column_names(df)
        
        # Validate columns
        is_valid, message = DataProcessor.validate_columns(df, required_cols)
        if not is_valid:
            st.error(message)
            return None
            
        # Convert demand to numeric
        if STANDARD_COLUMNS['demand'] in df.columns:
            try:
                df[STANDARD_COLUMNS['demand']] = pd.to_numeric(df[STANDARD_COLUMNS['demand']], errors='coerce')
                invalid_rows = df[STANDARD_COLUMNS['demand']].isna().sum()
                if invalid_rows > 0:
                    st.warning(f"Found {invalid_rows} non-numeric values in demand column. Rows removed.")
                    df = df.dropna(subset=[STANDARD_COLUMNS['demand']])
            except Exception as e:
                st.error(f"Error converting demand to numeric: {str(e)}")
                return None
        else:
            st.error(f"Required column '{STANDARD_COLUMNS['demand']}' not found.")
            return None
            
        with st.spinner("Preprocessing data..."):
            # Convert dates
            df = DataProcessor.convert_dates(df, list(set(date_cols)))
            # Handle missing values
            df = DataProcessor.handle_missing_values(df, fill_rules)
            # Add derived features
            df = DataProcessor.add_derived_features(df)
            # Drop duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
        st.session_state.state.processed_data = df
        return df

    @staticmethod
    def prepare_time_series(df: pd.DataFrame, 
                           target_col: str = STANDARD_COLUMNS['demand'], 
                           date_col: str = STANDARD_COLUMNS['date'], 
                           freq: str = 'ME') -> Optional[pd.Series]:
        """
        Convert DataFrame to time series format.
        """
        if df is None:
            st.error("Input DataFrame is None.")
            return None
            
        # Use processed data if available
        if hasattr(st.session_state.state, 'processed_data'):
            df = st.session_state.state.processed_data
        else:
            df = standardize_column_names(df)
            
        # Validate columns
        is_valid, message = DataProcessor.validate_columns(df, [date_col, target_col])
        if not is_valid:
            st.error(message)
            return None
            
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            st.error(f"Column '{date_col}' must be in datetime format.")
            return None
            
        try:
            with st.spinner("Preparing time series..."):
                ts = df.set_index(date_col)[target_col].resample(freq).sum()
                if ts.empty:
                    st.warning("Time series is empty after resampling.")
                    return None
                return ts
                
        except Exception as e:
            st.error(f"Error preparing time series: {str(e)}.")
            return None