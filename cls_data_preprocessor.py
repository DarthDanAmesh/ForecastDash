import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Tuple
import logging
from column_config import STANDARD_COLUMNS, standardize_column_names, COLUMN_ALIASES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def validate_columns(_df: pd.DataFrame, required_cols: List[str], 
                        optional_cols: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Validate the presence of required and optional columns in the DataFrame.
        
        Args:
            _df (pd.DataFrame): Input DataFrame.
            required_cols (List[str]): List of required column names.
            optional_cols (List[str], optional): List of optional column names.
        
        Returns:
            Tuple[bool, str]: (is_valid, message) indicating if columns are valid and any error/warning message.
        """
        try:
            if _df is None or _df.empty:
                return False, "Input DataFrame is None or empty. Please upload a valid CSV/Excel file."

            df = standardize_column_names(_df)
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                return False, (
                    f"Missing required columns: {', '.join(missing_required)}. "
                    "Please ensure your file includes these columns or supported aliases (e.g., 'quantity' for 'demand'). "
                    "Download a sample template from the error message below."
                )

            if optional_cols:
                missing_optional = [col for col in optional_cols if col not in df.columns]
                if missing_optional:
                    return True, (
                        f"Warning: Missing optional columns: {', '.join(missing_optional)}. "
                        "These columns enhance analysis (e.g., 'material' for discontinued products) but are not mandatory."
                    )
            return True, ""
        except Exception as e:
            logger.error(f"Column validation failed: {str(e)}")
            return False, f"Error validating columns: {str(e)}. Please check the file format."

    @staticmethod
    def convert_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        """
        Convert specified columns to datetime format if not already datetime.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            date_cols (List[str]): List of column names to convert to datetime.
        
        Returns:
            pd.DataFrame: DataFrame with converted date columns.
        """
        df = df.copy()
        for col in [c for c in date_cols if c in df.columns]:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].isna().all():
                        st.warning(
                            f"Column '{col}' could not be converted to datetime. All values are invalid. "
                            "Please use a format like YYYY-MM-DD."
                        )
                except Exception as e:
                    st.warning(
                        f"Error converting column '{col}' to datetime: {str(e)}. Values set to NaT. "
                        "Please check the date format in your file."
                    )
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
        for col, fill_value in fill_rules.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
        
        # Handle numeric columns with median
        if STANDARD_COLUMNS['demand'] in df.columns:
            df[STANDARD_COLUMNS['demand']] = df[STANDARD_COLUMNS['demand']].fillna(
                df[STANDARD_COLUMNS['demand']].median()
            )
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
        planned_date_col = STANDARD_COLUMNS['planned_delivery_date']
        actual_date_col = STANDARD_COLUMNS['delivery_date']
        
        if planned_date_col in df.columns and actual_date_col in df.columns:
            try:
                df['Delivery_Delay'] = (df[actual_date_col] - df[planned_date_col]).dt.days
                df['Delivery_Delay'] = df['Delivery_Delay'].fillna(0)
            except Exception as e:
                st.warning(
                    f"Error calculating Delivery_Delay: {str(e)}. Column not added. "
                    "Ensure date columns are in valid datetime format."
                )
        return df

    @staticmethod
    @st.cache_data(show_spinner=False)
    def preprocess_data(_df: pd.DataFrame, 
                       required_cols: List[str] = [
                           STANDARD_COLUMNS['date'], 
                           STANDARD_COLUMNS['demand']
                       ],
                       optional_cols: List[str] = [
                           STANDARD_COLUMNS['material'], 
                           STANDARD_COLUMNS['country'], 
                           STANDARD_COLUMNS['delivery_date'], 
                           STANDARD_COLUMNS['delivery_quantity'], 
                           STANDARD_COLUMNS['planned_delivery_date'], 
                           STANDARD_COLUMNS['order_date']
                       ],
                       date_cols: List[str] = [
                           STANDARD_COLUMNS['date'], 
                           STANDARD_COLUMNS['planned_delivery_date'], 
                           STANDARD_COLUMNS['delivery_date'], 
                           STANDARD_COLUMNS['order_date']
                       ],
                       fill_rules: Dict[str, str] = {
                           'Customer Reference': 'Unknown', 
                           'Customer Material': 'Unknown', 
                           STANDARD_COLUMNS['material']: 'Unknown', 
                           STANDARD_COLUMNS['country']: 'Unknown', 
                           STANDARD_COLUMNS['delivery_date']: pd.Timestamp('1970-01-01'), 
                           STANDARD_COLUMNS['delivery_quantity']: 0
                       }) -> Optional[pd.DataFrame]:
        """
        Clean and preprocess the input data.
        
        Args:
            _df (pd.DataFrame): Input DataFrame.
            required_cols (List[str]): List of required column names.
            optional_cols (List[str]): List of optional column names.
            date_cols (List[str]): List of columns to convert to datetime.
            fill_rules (Dict[str, str]): Dictionary mapping columns to fill values.
        
        Returns:
            Optional[pd.DataFrame]: Preprocessed DataFrame or None if validation fails.
        """
        try:
            if _df is None or _df.empty:
                st.error("No data provided. Please upload a valid CSV/Excel file.")
                return None

            df = _df.copy()
            required_cols = list(set(required_cols))
            date_cols = list(set(date_cols))

            # Standardize column names
            df = standardize_column_names(df)
            column_case_mapping = {col.lower(): col for col in df.columns}

            # Validate columns
            is_valid, message = DataProcessor.validate_columns(df, required_cols, optional_cols)
            if not is_valid:
                st.error(message, icon="🚨")
                return None
            if message:
                st.warning(message, icon="⚠️")

            # Ensure demand is numeric
            if STANDARD_COLUMNS['demand'] in df.columns:
                try:
                    df[STANDARD_COLUMNS['demand']] = pd.to_numeric(
                        df[STANDARD_COLUMNS['demand']], errors='coerce'
                    )
                    invalid_rows = df[STANDARD_COLUMNS['demand']].isna().sum()
                    if invalid_rows > 0:
                        st.warning(
                            f"{invalid_rows} non-numeric values in 'demand' converted to NaN and filled with median."
                        )
                except Exception as e:
                    st.error(
                        f"Error converting demand to numeric: {str(e)}. Please ensure demand values are numbers."
                    )
                    return None
            else:
                st.error(f"Required column '{STANDARD_COLUMNS['demand']}' not found.")
                return None

            with st.spinner("Preprocessing data..."):
                # Convert dates
                df = DataProcessor.convert_dates(df, date_cols)
                # Handle missing values
                df = DataProcessor.handle_missing_values(df, fill_rules)
                # Add derived features
                df = DataProcessor.add_derived_features(df)
                # Drop duplicates
                df = df.loc[:, ~df.columns.duplicated()]

            # Restore original case for display
            reverse_case_mapping = {v: k for k, v in column_case_mapping.items()}
            display_df = df.rename(columns=reverse_case_mapping)

            # Store processed data
            st.session_state.state.processed_data = df
            return display_df
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            st.error(
                f"Error preprocessing data: {str(e)}. Please check the file format and column names."
            )
            return None

    @staticmethod
    @st.cache_data(show_spinner=False)
    def prepare_time_series(_df: pd.DataFrame, 
                           target_col: str = STANDARD_COLUMNS['demand'], 
                           date_col: str = STANDARD_COLUMNS['date'], 
                           freq: str = 'ME') -> Optional[pd.Series]:
        """
        Convert DataFrame to time series format.
        
        Args:
            _df (pd.DataFrame): Input DataFrame.
            target_col (str): Column to use as the time series values.
            date_col (str): Column to use as the time index.
            freq (str): Resampling frequency.
        
        Returns:
            Optional[pd.Series]: Resampled time series or None if validation fails.
        """
        try:
            if _df is None or _df.empty:
                st.error("No data provided for time series. Please upload a valid dataset.")
                return None

            df = _df.copy()
            is_valid, message = DataProcessor.validate_columns(df, [date_col, target_col])
            if not is_valid:
                st.error(message)
                return None

            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                st.error(
                    f"Column '{date_col}' is not in datetime format. "
                    "Please ensure dates are valid (e.g., YYYY-MM-DD)."
                )
                return None

            if not pd.api.types.is_numeric_dtype(df[target_col]):
                try:
                    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                    if df[target_col].isna().any():
                        st.warning(
                            f"Non-numeric values in '{target_col}' converted to NaN and dropped. "
                            "Ensure demand values are numbers."
                        )
                        df = df.dropna(subset=[target_col])
                except Exception as e:
                    st.error(f"Error converting '{target_col}' to numeric: {str(e)}.")
                    return None

            with st.spinner("Preparing time series..."):
                ts = df.set_index(date_col)[target_col].resample(freq).sum()

                if ts.empty:
                    st.warning(
                        "Time series is empty after resampling. "
                        "Check if the date range is sufficient or adjust the frequency (e.g., 'ME' for monthly)."
                    )
                    return None

                if ts.isna().all():
                    st.error(
                        f"All values in '{target_col}' are invalid after resampling. "
                        "Please verify the data."
                    )
                    return None

                return ts
        except Exception as e:
            logger.error(f"Time series preparation failed: {str(e)}")
            st.error(
                f"Error preparing time series: {str(e)}. "
                "Please verify the date and demand columns."
            )
            return None