# cls_data_preprocessor.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
from constants import STANDARD_COLUMNS, COLUMN_ALIASES, DEFAULT_FREQ, MIN_OBSERVATIONS_PER_MATERIAL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def validate_columns(_df: pd.DataFrame, required_cols: List[str], 
                        optional_cols: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Validate the presence of required and optional columns in the DataFrame."""
        try:
            if _df is None or _df.empty:
                return False, "Input DataFrame is None or empty. Please upload a valid CSV file."
            
            df = DataProcessor.standardize_column_names(_df)
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                return False, (
                    f"Missing required columns: {', '.join(missing_required)}. "
                    "Ensure your file includes these columns or supported aliases (e.g., 'quantity' for 'demand'). "
                    "Download a sample template from the error message."
                )
            
            if optional_cols:
                missing_optional = [col for col in optional_cols if col not in df.columns]
                if missing_optional:
                    return True, (
                        f"Warning: Missing optional columns: {', '.join(missing_optional)}. "
                        "These enhance analysis (e.g., 'material' for SKU-level forecasts) but are not mandatory."
                    )
            return True, ""
        except Exception as e:
            logger.error(f"Column validation failed: {str(e)}")
            return False, f"Error validating columns: {str(e)}. Check file format."

    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names based on COLUMN_ALIASES."""
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        column_mapping = {}
        used_std_keys = set()
        alias_lookup = {}
        for std_key, aliases in COLUMN_ALIASES.items():
            for i, alias in enumerate(aliases):
                alias_lookup[alias.lower()] = (std_key, i)
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in alias_lookup:
                std_key, priority = alias_lookup[col_lower]
                if std_key not in used_std_keys:
                    column_mapping[col] = std_key
                    used_std_keys.add(std_key)
                elif std_key in column_mapping.values():
                    current_col = next(k for k, v in column_mapping.items() if v == std_key)
                    current_priority = alias_lookup[current_col.lower()][1]
                    if priority < current_priority:
                        column_mapping[col] = std_key
                        del column_mapping[current_col]
        df = df.rename(columns=column_mapping)
        logger.info(f"Standardized columns: {list(df.columns)}")
        return df

    @staticmethod
    def convert_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        """Convert specified columns to datetime with multiple format attempts."""
        df = df.copy()
        date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d.%m.%Y', '%d-%m-%Y', '%Y/%m/%d']
        for col in [c for c in date_cols if c in df.columns]:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                original_values = df[col].copy()
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                invalid_count = df[col].isna().sum()
                if invalid_count > 0:
                    for fmt in date_formats:
                        try:
                            temp = pd.to_datetime(original_values, format=fmt, errors='coerce')
                            if temp.isna().sum() < invalid_count:
                                df[col] = temp
                                invalid_count = temp.isna().sum()
                                logger.info(f"Parsed '{col}' with format {fmt}, {invalid_count} invalid dates remain")
                                break
                        except:
                            continue
                    if invalid_count > 0:
                        st.warning(f"{invalid_count} invalid dates in '{col}' after trying all formats.", icon="⚠️")
        return df

    @staticmethod
    def clean_numeric_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Clean and convert a column to numeric, handling commas and invalid values."""
        df = df.copy()
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = df[col].str.replace(r'[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col == STANDARD_COLUMNS['demand']:
                    df[col] = df[col].abs()
                    df[col] = df[col].replace(0, 1e-6)
                invalid_count = df[col].isna().sum()
                if invalid_count > 0:
                    st.warning(f"{invalid_count} invalid numeric values in '{col}' set to NaN.", icon="⚠️")
            except Exception as e:
                logger.error(f"Error cleaning numeric column '{col}': {str(e)}")
                st.error(f"Error cleaning '{col}': {str(e)}.", icon="🚨")
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, fill_rules: Dict[str, str]) -> pd.DataFrame:
        """Handle missing values according to specified rules."""
        df = df.copy()
        for col, fill_value in fill_rules.items():
            if col in df.columns:
                if fill_value == 'median':
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df = DataProcessor.clean_numeric_column(df, col)
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(fill_value)
        return df

    @staticmethod
    def create_global_time_idx(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create a global time index across all groups."""
        df = df.copy()
        unique_dates = sorted(df[date_col].unique())
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        df['time_idx'] = df[date_col].map(date_to_idx)
        df['time_idx'] = df['time_idx'].astype('int64')
        logger.info(f"Created global time_idx with range 0 to {df['time_idx'].max()}")
        return df

    @staticmethod
    def resample_and_aggregate(df: pd.DataFrame, group_col: str, date_col: str, freq: str = DEFAULT_FREQ) -> pd.DataFrame:
        """Resample data to specified frequency and aggregate by group."""
        df = df.copy()
        df_indexed = df.set_index(date_col)
        resampled_list = []
        for material, group in df_indexed.groupby(group_col):
            resampled_group = group.resample(freq).agg({
                'demand': 'sum',
                'country': 'first',
                'month': 'first',
                'day_of_week': 'first',
                'is_weekend': 'first',
                'quarter': 'first',
                'year': 'first',
                'delivery_delay': 'mean'
            }).reset_index()
            resampled_group[group_col] = material
            resampled_group = resampled_group[resampled_group['demand'] > 0]
            if not resampled_group.empty:
                resampled_list.append(resampled_group)
        if not resampled_list:
            logger.error("No data remained after resampling")
            return pd.DataFrame()
        result = pd.concat(resampled_list, ignore_index=True)
        result = result.sort_values([group_col, date_col])
        result = DataProcessor.create_global_time_idx(result, date_col)
        logger.info(f"Resampled data to {freq} frequency, final shape: {result.shape}")
        return result

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
                           STANDARD_COLUMNS['material']: 'Unknown',
                           STANDARD_COLUMNS['country']: 'Unknown',
                           STANDARD_COLUMNS['demand']: 'median',
                           STANDARD_COLUMNS['delivery_date']: pd.Timestamp('1970-01-01'),
                           STANDARD_COLUMNS['delivery_quantity']: 0
                       }) -> Optional[pd.DataFrame]:
        """Full preprocessing pipeline for modeling."""
        try:
            if _df is None or _df.empty:
                st.error("No data provided. Upload a valid CSV file.", icon="🚨")
                return None

            df = _df.copy()
            logger.info(f"Initial data shape: {df.shape}")

            # Standardize column names
            df = DataProcessor.standardize_column_names(df)

            # Validate columns
            is_valid, message = DataProcessor.validate_columns(df, required_cols, optional_cols)
            if not is_valid:
                st.error(message, icon="🚨")
                return None
            if message:
                st.warning(message, icon="⚠️")

            # Clean demand column
            df = DataProcessor.clean_numeric_column(df, STANDARD_COLUMNS['demand'])
            initial_rows = len(df)
            df = df.dropna(subset=[STANDARD_COLUMNS['demand']])
            df = df[df[STANDARD_COLUMNS['demand']] > 0]
            logger.info(f"Removed {initial_rows - len(df)} rows with invalid/zero demand")

            if df.empty:
                st.error("No valid demand data remaining.", icon="🚨")
                return None

            # Convert dates
            df = DataProcessor.convert_dates(df, date_cols)
            date_col = STANDARD_COLUMNS['date']
            initial_rows = len(df)
            df = df.dropna(subset=[date_col])
            logger.info(f"Removed {initial_rows - len(df)} rows with invalid dates")

            if df.empty:
                st.error("No valid date data remaining.", icon="🚨")
                return None

            # Handle missing values
            df = DataProcessor.handle_missing_values(df, fill_rules)

            # Filter materials with sufficient observations
            if STANDARD_COLUMNS['material'] in df.columns:
                material_counts = df[STANDARD_COLUMNS['material']].value_counts()
                valid_materials = material_counts[material_counts >= MIN_OBSERVATIONS_PER_MATERIAL].index
                initial_materials = df[STANDARD_COLUMNS['material']].nunique()
                df = df[df[STANDARD_COLUMNS['material']].isin(valid_materials)]
                final_materials = df[STANDARD_COLUMNS['material']].nunique()
                logger.info(f"Filtered materials: {initial_materials} -> {final_materials} (min {MIN_OBSERVATIONS_PER_MATERIAL} observations)")
                if df.empty:
                    st.error("No materials with sufficient data points.", icon="🚨")
                    return None

            # Add derived features (from funct_feature_eng)
            from funct_feature_eng import enhance_feature_engineering
            df = enhance_feature_engineering(df)

            # Resample and aggregate
            if STANDARD_COLUMNS['material'] in df.columns:
                df = DataProcessor.resample_and_aggregate(df, STANDARD_COLUMNS['material'], STANDARD_COLUMNS['date'])
                if df.empty:
                    st.error("No data remaining after resampling.", icon="🚨")
                    return None

            logger.info(f"Final preprocessing shape: {df.shape}")
            logger.info(f"Materials: {df[STANDARD_COLUMNS['material']].nunique()}")
            logger.info(f"Date range: {df[STANDARD_COLUMNS['date']].min()} to {df[STANDARD_COLUMNS['date']].max()}")
            logger.info(f"Time index range: {df['time_idx'].min()} to {df['time_idx'].max()}")
            st.session_state.state.processed_data = df
            return df
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            st.error(f"Preprocessing failed: {str(e)}.", icon="🚨")
            return None

    @staticmethod
    @st.cache_data(show_spinner=False)
    def prepare_time_series(_df: pd.DataFrame, 
                           target_col: str = STANDARD_COLUMNS['demand'], 
                           date_col: str = STANDARD_COLUMNS['date'], 
                           freq: str = DEFAULT_FREQ) -> Optional[pd.DataFrame]:
        """Prepare time series data for analysis (e.g., anomaly detection)."""
        try:
            if _df is None or _df.empty:
                st.error("No data provided for time series. Upload a valid dataset.", icon="🚨")
                return None

            df = _df.copy()
            is_valid, message = DataProcessor.validate_columns(df, [date_col, target_col])
            if not is_valid:
                st.error(message, icon="🚨")
                return None

            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                st.error(f"Column '{date_col}' is not in datetime format. Ensure dates are valid (e.g., YYYY-MM-DD).", icon="🚨")
                return None

            if not pd.api.types.is_numeric_dtype(df[target_col]):
                df = DataProcessor.clean_numeric_column(df, target_col)
                if df[target_col].isna().any():
                    st.warning(f"Non-numeric values in '{target_col}' converted to NaN and dropped.", icon="⚠️")
                    df = df.dropna(subset=[target_col])

            with st.spinner("Preparing time series..."):
                if STANDARD_COLUMNS['material'] in df.columns:
                    # Group by material and date, sum demand
                    ts = df.groupby([STANDARD_COLUMNS['material'], date_col])[target_col].sum().reset_index()
                    ts = ts.pivot(index=date_col, columns=STANDARD_COLUMNS['material'], values=target_col)
                    ts = ts.resample(freq).sum()
                else:
                    # Aggregate without material
                    ts = df.set_index(date_col)[target_col].resample(freq).sum().to_frame()

                if ts.empty:
                    st.warning("Time series empty after resampling. Check date range or frequency.", icon="⚠️")
                    return None

                if ts.isna().all().all():
                    st.error("All values in time series are invalid. Verify data.", icon="🚨")
                    return None

                return ts
        except Exception as e:
            logger.error(f"Time series preparation failed: {str(e)}")
            st.error(f"Error preparing time series: {str(e)}.", icon="🚨")
            return None