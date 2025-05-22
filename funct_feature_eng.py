# funct_feature_eng.py
import pandas as pd
import streamlit as st
import logging
from constants import STANDARD_COLUMNS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhance_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for modeling, aligned with DeepAR and XGBoost requirements."""
    if df is None or len(df) == 0:
        logger.warning("Empty dataframe passed to enhance_feature_engineering")
        st.warning("Empty dataframe provided. No features added.", icon="⚠️")
        return df

    df = df.copy()
    logger.info(f"Original columns: {df.columns.tolist()}")

    # Date-based features
    date_col = STANDARD_COLUMNS['date']
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df['month'] = df[date_col].dt.month
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
            df['quarter'] = df[date_col].dt.quarter
            df['year'] = df[date_col].dt.year
            df['week_of_year'] = df[date_col].dt.isocalendar().week
            logger.info("Added date-based features")
        except Exception as e:
            logger.error(f"Error creating date features: {str(e)}")
            st.warning("Error creating date-based features.", icon="⚠️")
    else:
        logger.warning(f"No valid datetime column '{date_col}' found")
        st.warning(f"Column '{date_col}' missing or not in datetime format.", icon="⚠️")

    # Delivery delay feature
    planned_date_col = STANDARD_COLUMNS['planned_delivery_date']
    delivery_date_col = STANDARD_COLUMNS['delivery_date']
    if planned_date_col in df.columns and delivery_date_col in df.columns:
        try:
            df['delivery_delay'] = (df[delivery_date_col] - df[planned_date_col]).dt.days
            df['delivery_delay'] = df['delivery_delay'].fillna(0)
            logger.info("Added delivery_delay feature")
        except Exception as e:
            logger.error(f"Error calculating delivery_delay: {str(e)}")
            st.warning("Error calculating delivery delay.", icon="⚠️")
    else:
        df['delivery_delay'] = 0
        logger.info("Set delivery_delay to 0 (missing date columns)")

    logger.info(f"Added features: {set(df.columns) - set(df.columns)}")
    return df