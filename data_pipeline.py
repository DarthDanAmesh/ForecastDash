# data_pipeline.py
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standardized column configuration
STANDARD_COLUMNS = {
    'date': 'date',
    'demand': 'demand',
    'material': 'material',
    'country': 'country',
    'delivery_date': 'delivery_date',
    'planned_delivery_date': 'planned_delivery_date',
    'order_date': 'order_date',
    'delivery_quantity': 'delivery_quantity',
}

COLUMN_ALIASES = {
    'date': ['date', 'created on', 'Created On', 'customer ref. date', 'Customer Ref. Date'],
    'demand': ['deliv.value lips doc.curr.', 'value', 'Delivery Quantity', 'delivery quantity'],
    'material': ['material', 'Material', 'customer material', 'Customer Material'],
    'country': ['country', 'Country Key Ship-to', 'country key ship-to'],
    'delivery_date': ['act. gds issue date', 'Act. Gds Issue Date'],
    'planned_delivery_date': ['pland gds mvmnt date', 'Pland Gds Mvmnt Date'],
    'order_date': ['order date', 'Customer Ref. Date'],
    'delivery_quantity': ['delivery quantity', 'Delivery Quantity'],
}

class DataPipeline:
    @staticmethod
    def load_csv(file_path: str) -> Optional[pd.DataFrame]:
        """Load and validate the CSV file."""
        try:
            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            if df.empty:
                logger.error("CSV file is empty.")
                try:
                    st.error("Uploaded CSV file is empty.", icon="🚨")
                except Exception:
                    logger.warning("Streamlit context missing, skipping UI error display")
                return None
            logger.info(f"Loaded CSV with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            try:
                st.error(f"Error loading CSV: {str(e)}", icon="🚨")
            except Exception:
                logger.warning("Streamlit context missing, skipping UI error display")
            return None

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
        """Convert specified columns to datetime format with multiple format attempts."""
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
                        logger.warning(f"{invalid_count} invalid dates in '{col}' after trying all formats")
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
                if col == 'demand':
                    df[col] = df[col].abs()
                    df[col] = df[col].replace(0, 1e-6)
                invalid_count = df[col].isna().sum()
                if invalid_count > 0:
                    logger.warning(f"{invalid_count} invalid numeric values in '{col}' set to NaN")
            except Exception as e:
                logger.error(f"Error cleaning numeric column '{col}': {str(e)}")
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, fill_rules: Dict[str, str]) -> pd.DataFrame:
        """Handle missing values according to specified rules."""
        df = df.copy()
        for col, fill_value in fill_rules.items():
            if col in df.columns:
                if fill_value == 'median':
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df = DataPipeline.clean_numeric_column(df, col)
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(fill_value)
        return df

    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for DeepAR."""
        df = df.copy()
        date_col = STANDARD_COLUMNS['date']
        if date_col in df.columns:
            df['month'] = df[date_col].dt.month
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['quarter'] = df[date_col].dt.quarter
            df['year'] = df[date_col].dt.year
            df['week_of_year'] = df[date_col].dt.isocalendar().week
            logger.info("Added date-based features")
        planned_date_col = STANDARD_COLUMNS['planned_delivery_date']
        delivery_date_col = STANDARD_COLUMNS['delivery_date']
        if planned_date_col in df.columns and delivery_date_col in df.columns:
            df['delivery_delay'] = (df[delivery_date_col] - df[planned_date_col]).dt.days
            df['delivery_delay'] = df['delivery_delay'].fillna(0)
            logger.info("Added delivery_delay feature")
        else:
            df['delivery_delay'] = 0
        return df

    @staticmethod
    def create_global_time_idx(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create a global time index that works across all groups."""
        df = df.copy()
        unique_dates = sorted(df[date_col].unique())
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        df['time_idx'] = df[date_col].map(date_to_idx)
        df['time_idx'] = df['time_idx'].astype('int64')
        logger.info(f"Created global time_idx with range 0 to {df['time_idx'].max()}")
        return df

    @staticmethod
    def resample_data(df: pd.DataFrame, group_col: str, date_col: str, freq: str = 'W') -> pd.DataFrame:
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
                'week_of_year': 'first',
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
        result = DataPipeline.create_global_time_idx(result, date_col)
        logger.info(f"Resampled data to {freq} frequency, shape: {result.shape}")
        return result

    @staticmethod
    def preprocess_data(file_path: str, freq: str = 'W') -> Optional[pd.DataFrame]:
        """Full preprocessing pipeline for DeepAR modeling."""
        try:
            # Step 1: Load CSV
            df = DataPipeline.load_csv(file_path)
            if df is None:
                return None
            logger.info(f"Initial data shape: {df.shape}")

            # Step 2: Standardize column names
            df = DataPipeline.standardize_column_names(df)

            # Step 3: Check for required columns
            required_cols = [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['material']]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                try:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}", icon="🚨")
                except Exception:
                    logger.warning("Streamlit context missing, skipping UI error display")
                return None

            # Step 4: Clean demand column
            df = DataPipeline.clean_numeric_column(df, STANDARD_COLUMNS['demand'])
            initial_rows = len(df)
            df = df.dropna(subset=[STANDARD_COLUMNS['demand']])
            df = df[df[STANDARD_COLUMNS['demand']] > 0]
            logger.info(f"Removed {initial_rows - len(df)} rows with invalid/zero demand")
            if df.empty:
                logger.error("No valid demand data remaining")
                return None

            # Step 5: Convert date columns
            date_cols = [col for col in [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['planned_delivery_date'], 
                         STANDARD_COLUMNS['delivery_date'], STANDARD_COLUMNS['order_date']] if col in df.columns]
            df = DataPipeline.convert_dates(df, date_cols)

            # Step 6: Handle missing dates
            date_col = STANDARD_COLUMNS['date']
            initial_rows = len(df)
            df = df.dropna(subset=[date_col])
            logger.info(f"Removed {initial_rows - len(df)} rows with invalid dates")
            if df.empty:
                logger.error("No valid date data remaining")
                return None

            # Step 7: Handle missing values
            fill_rules = {
                STANDARD_COLUMNS['material']: 'Unknown',
                STANDARD_COLUMNS['country']: 'Unknown',
                STANDARD_COLUMNS['demand']: 'median'
            }
            df = DataPipeline.handle_missing_values(df, fill_rules)

            # Step 8: Filter materials with sufficient data
            material_counts = df[STANDARD_COLUMNS['material']].value_counts()
            min_observations = 10
            valid_materials = material_counts[material_counts >= min_observations].index
            initial_materials = df[STANDARD_COLUMNS['material']].nunique()
            df = df[df[STANDARD_COLUMNS['material']].isin(valid_materials)]
            logger.info(f"Filtered materials: {initial_materials} -> {df[STANDARD_COLUMNS['material']].nunique()} (min {min_observations} observations)")
            if df.empty:
                logger.error("No materials with sufficient data points")
                return None

            # Step 9: Add derived features
            df = DataPipeline.add_derived_features(df)

            # Step 10: Resample to specified frequency
            df = DataPipeline.resample_data(df, STANDARD_COLUMNS['material'], STANDARD_COLUMNS['date'], freq)
            if df.empty:
                logger.error("No data remaining after resampling")
                return None

            # Step 11: Validate time_idx
            if not pd.api.types.is_integer_dtype(df['time_idx']):
                logger.warning("time_idx is not integer type, converting to int64")
                df['time_idx'] = df['time_idx'].astype('int64')

            # Step 12: Final validation
            logger.info(f"Final preprocessing shape: {df.shape}")
            logger.info(f"Materials: {df[STANDARD_COLUMNS['material']].nunique()}")
            logger.info(f"Date range: {df[STANDARD_COLUMNS['date']].min()} to {df[STANDARD_COLUMNS['date']].max()}")
            logger.info(f"Time index range: {df['time_idx'].min()} to {df['time_idx'].max()}")
            return df

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            try:
                st.error(f"Preprocessing failed: {str(e)}", icon="🚨")
            except Exception:
                logger.warning("Streamlit context missing, skipping UI error display")
            return None