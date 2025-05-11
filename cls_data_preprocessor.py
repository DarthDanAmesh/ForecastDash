# cls_data_preprocessor.py
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
from column_config import STANDARD_COLUMNS, standardize_column_names, COLUMN_ALIASES

class DataProcessor:
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: List[str], optional_cols: List[str] = None) -> tuple[bool, str]:
        """
        Validate the presence of required and optional columns in the DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame.
            required_cols (List[str]): List of required column names.
            optional_cols (List[str], optional): List of optional column names.
        Returns:
            tuple[bool, str]: (is_valid, message) indicating if columns are valid and any error/warning message.
        """
        if df is None:
            return False, "Input DataFrame is None."
        
        # Standardize column names before validation
        df = standardize_column_names(df)
        
        # Check for required columns
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            return False, f"Missing required columns: {', '.join(missing_required)}."
            
        # Check for optional columns
        if optional_cols:
            missing_optional = [col for col in optional_cols if col not in df.columns]
            if missing_optional:
                return True, f"Warning: Missing optional columns: {', '.join(missing_optional)}."
                
        return True, ""

    @staticmethod
    def convert_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        """
        Convert specified columns to datetime format.
        Args:
            df (pd.DataFrame): Input DataFrame.
            date_cols (List[str]): List of column names to convert to datetime.
        Returns:
            pd.DataFrame: DataFrame with converted date columns.
        """
        df = df.copy()
        
        # Only keep date columns that exist in the dataframe
        existing_date_cols = [col for col in date_cols if col in df.columns]
        
        for col in existing_date_cols:
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
        Args:
            df (pd.DataFrame): Input DataFrame.
            fill_rules (Dict[str, str]): Dictionary mapping column names to fill values.
        Returns:
            pd.DataFrame: DataFrame with handled missing values.
        """
        df = df.copy()
        
        # Only apply fill rules to columns that exist in the dataframe
        for col, fill_value in fill_rules.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
                
        return df

    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features such as Delivery_Delay.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with added features.
        """
        df = df.copy()
        
        # Use standardized column names for dependency checking
        planned_date_col = STANDARD_COLUMNS['planned_delivery_date']
        actual_date_col = STANDARD_COLUMNS['delivery_date']
        
        # Create delivery delay feature if both dates are present
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
                       fill_rules: Dict[str, str] = {'Customer Reference': 'Unknown', 
                                                     'Customer Material': 'Unknown'}) -> Optional[pd.DataFrame]:
        """
        Clean and preprocess the input data.
        Args:
            df (pd.DataFrame): Input DataFrame.
            required_cols (List[str]): List of required column names.
            date_cols (List[str]): List of columns to convert to datetime.
            fill_rules (Dict[str, str]): Dictionary mapping columns to fill values for missing data.
        Returns:
            Optional[pd.DataFrame]: Preprocessed DataFrame or None if validation fails.
        """
        if df is None:
            st.error("Input DataFrame is None. Please provide valid data.")
            return None
            
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Remove duplicates from required/date cols
        required_cols = list(set(required_cols))
        date_cols = list(set(date_cols))
        
        # Standardize column names once and only once
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        # Create mapping from lowercase to original case for later restoration
        column_case_mapping = {col.lower(): col for col in df.columns}
        
        # Apply column aliases mapping to lowercase names
        inverse_alias_map = {}
        for std_name, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                inverse_alias_map[alias.lower()] = std_name
                
        # Apply mappings
        new_columns = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in inverse_alias_map:
                new_columns[col] = inverse_alias_map[col_lower]
            else:
                new_columns[col] = col_lower
                
        df = df.rename(columns=new_columns)
        
        # Now ensure we have all required columns
        is_valid, message = DataProcessor.validate_columns(df, required_cols)
        if not is_valid:
            st.error(message)
            return None
            
        # Ensure demand column is numeric
        if STANDARD_COLUMNS['demand'] in df.columns:
            try:
                df[STANDARD_COLUMNS['demand']] = pd.to_numeric(df[STANDARD_COLUMNS['demand']], errors='coerce')
                invalid_rows = df[STANDARD_COLUMNS['demand']].isna().sum()
                if invalid_rows > 0:
                    st.warning(f"Found {invalid_rows} non-numeric values in demand column that were converted to NaN and will be removed")
                    df = df.dropna(subset=[STANDARD_COLUMNS['demand']])
            except Exception as e:
                st.error(f"Error converting demand to numeric: {str(e)}")
                return None
        else:
            st.error(f"Required column '{STANDARD_COLUMNS['demand']}' not found after standardization")
            return None
            
        # Preprocessing steps
        with st.spinner("Preprocessing data..."):
            # Convert dates
            df = DataProcessor.convert_dates(df, date_cols)
            # Handle missing values
            df = DataProcessor.handle_missing_values(df, fill_rules)
            # Add derived features
            df = DataProcessor.add_derived_features(df)
            
        # Drop duplicate columns if any exist
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Restore original case for display purposes
        reverse_case_mapping = {v: k for k, v in column_case_mapping.items()}
        display_df = df.rename(columns=reverse_case_mapping)
        
        # Store the processed data in session state but return the display version
        st.session_state.state.processed_data = df  # Store internal version
        return display_df

    @staticmethod
    def prepare_time_series(df: pd.DataFrame, 
                           target_col: str = STANDARD_COLUMNS['demand'], 
                           date_col: str = STANDARD_COLUMNS['date'], 
                           freq: str = 'ME') -> Optional[pd.Series]:
        """
        Convert DataFrame to time series format.
        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Column to use as the time series values.
            date_col (str): Column to use as the time index.
            freq (str): Resampling frequency.
        Returns:
            Optional[pd.Series]: Resampled time series or None if validation fails.
        """
        if df is None:
            st.error("Input DataFrame is None. Please provide valid data.")
            return None
            
        # Use the processed data from session state if available
        if hasattr(st.session_state.state, 'processed_data'):
            df = st.session_state.state.processed_data
        else:
            # Standardize column names
            df = standardize_column_names(df)
            
        # Validate columns
        is_valid, message = DataProcessor.validate_columns(df, [date_col, target_col])
        if not is_valid:
            st.error(message)
            return None
            
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            st.error(f"Column '{date_col}' must be in datetime format. Please preprocess the data.")
            return None
            
        # Ensure target column is numeric
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            try:
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                if df[target_col].isna().any().any():
                    st.warning(f"Some values in '{target_col}' couldn't be converted to numbers. Rows with invalid values will be removed.")
                    df = df.dropna(subset=[target_col])
            except Exception as e:
                st.error(f"Error converting target column to numeric: {str(e)}")
                return None
                
        try:
            with st.spinner("Preparing time series..."):
                ts = df.set_index(date_col)[target_col]
                ts = ts.resample(freq).sum()
                
                if ts.empty:
                    st.warning("Time series is empty after resampling. Please check the data or adjust the frequency.")
                    return None
                    
                return ts
                
        except Exception as e:
            st.error(f"Error preparing time series: {str(e)}. Please verify the data and parameters.")
            return None