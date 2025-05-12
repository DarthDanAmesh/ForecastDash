import pandas as pd
import streamlit as st
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhance_feature_engineering(df):
    """
    Create more advanced features to improve forecast accuracy.
    
    This function is designed to be robust against various column naming conventions
    and will adapt to the available columns in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with at least a date column
        
    Returns:
        pd.DataFrame: Enhanced dataframe with additional features
    """
    if df is None or len(df) == 0:
        logger.warning("Empty dataframe passed to enhance_feature_engineering")
        return df
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Log the original columns for debugging
    logger.info(f"Original columns: {df.columns.tolist()}")
    
    # Identify the date column - handle multiple possible names
    date_columns = ['date', 'created on', 'created_on', 'timestamp', 'order date']
    date_col = None
    
    # Find the first matching date column
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    # Create date-based features if we have a date column
    if date_col:
        logger.info(f"Using '{date_col}' as date column")
        
        # Ensure the date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                logger.info(f"Converted '{date_col}' to datetime")
            except Exception as e:
                logger.error(f"Failed to convert '{date_col}' to datetime: {str(e)}")
                st.warning(f"Could not convert {date_col} to datetime. Some features won't be created.")
        
        # If conversion was successful, create date features
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                logger.info("Creating date-based features")
                df['month'] = df[date_col].dt.month
                df['quarter'] = df[date_col].dt.quarter
                df['year'] = df[date_col].dt.year
                df['day_of_week'] = df[date_col].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
                df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
                df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
                df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
                df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
            except Exception as e:
                logger.error(f"Error creating date features: {str(e)}")
                st.warning("Error creating date-based features. Processing will continue with limited features.")
    else:
        logger.warning(f"No date column found. Tried: {date_columns}")
        st.warning(f"No date column found. Please make sure your data includes one of: {', '.join(date_columns)}")
    
    # Create regional indicators (with fallback)
    country_columns = ['country key ship-to', 'country_key', 'country', 'region', 'country_code']
    country_col = None
    
    # Find the first matching country column
    for col in country_columns:
        if col in df.columns:
            country_col = col
            break
    
    if country_col:
        logger.info(f"Using '{country_col}' as country column")
        try:
            # Map country codes to regions
            df['region'] = df[country_col].astype(str).map(
                lambda x: 'Italy' if x.upper() in ('IT', 'ITA', 'ITALY') else 'Other')
            logger.info("Created region feature")
        except Exception as e:
            logger.error(f"Error creating region feature: {str(e)}")
    else:
        logger.warning(f"No country column found. Tried: {country_columns}")
    
    # Create product category features (with fallback)
    category_columns = ['material group', 'material_group', 'product_category', 'category', 'product_type']
    category_col = None
    
    # Find the first matching category column
    for col in category_columns:
        if col in df.columns:
            category_col = col
            break
    
    if category_col:
        logger.info(f"Using '{category_col}' as product category column")
        try:
            df['product_category'] = df[category_col].astype(str)
            logger.info("Created product_category feature")
        except Exception as e:
            logger.error(f"Error creating product category feature: {str(e)}")
    else:
        logger.warning(f"No product category column found. Tried: {category_columns}")
    
    # Order vs delivery gap analysis
    order_date_cols = ['customer ref. date', 'customer_ref_date', 'order_date']
    delivery_date_cols = ['act. gds issue date', 'delivery_date', 'shipment_date']
    
    order_date_col = None
    delivery_date_col = None
    
    # Find matching order date column
    for col in order_date_cols:
        if col in df.columns:
            order_date_col = col
            break
            
    # Find matching delivery date column
    for col in delivery_date_cols:
        if col in df.columns:
            delivery_date_col = col
            break
    
    if order_date_col and delivery_date_col:
        logger.info(f"Using '{order_date_col}' and '{delivery_date_col}' for delivery gap analysis")
        try:
            # Ensure both columns are datetime, the infer_datetime_format=True is used for faster parsing for non-standard date formats
            df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
            df[delivery_date_col] = pd.to_datetime(df[delivery_date_col], errors='coerce')
            
            # Calculate days between order and delivery with validation
            valid_dates_mask = df[order_date_col].notna() & df[delivery_date_col].notna()
            df['order_to_delivery_days'] = np.where(
                valid_dates_mask,
                (df[delivery_date_col] - df[order_date_col]).dt.days,
                np.nan
            )
            
            # Additional validation for negative values
            negative_mask = df['order_to_delivery_days'] < 0
            if negative_mask.any():
                logger.warning(f"{negative_mask.sum()} rows have negative order-to-delivery days")
                st.warning(f"Found {negative_mask.sum()} rows where delivery occurred before order.")
                df.loc[negative_mask, 'order_to_delivery_days'] = np.nan
            
            # Report invalid calculations
            invalid_count = df['order_to_delivery_days'].isna().sum()
            if invalid_count > 0:
                logger.warning(f"{invalid_count} rows have invalid order-to-delivery days")
                # Optionally log first few invalid rows for debugging
                invalid_rows = df[df['order_to_delivery_days'].isna()][[order_date_col, delivery_date_col]].head(2)
                logger.debug(f"Sample invalid rows:\n{invalid_rows}")
            
            logger.info("Created order_to_delivery_days feature")
        except Exception as e:
            logger.error(f"Error calculating order to delivery days: {str(e)}")
    else:
        logger.warning("Could not find matching order date and delivery date columns for gap analysis")
    
    # Sales organization performance
    sales_org_cols = ['sales organization', 'sales_org', 'sales_organization', 'org', 'business_unit']
    sales_org_col = None
    
    # Find matching sales org column
    for col in sales_org_cols:
        if col in df.columns:
            sales_org_col = col
            break
    
    if sales_org_col:
        logger.info(f"Using '{sales_org_col}' as sales organization column")
        try:
            df['sales_org'] = df[sales_org_col].astype(str)
            logger.info("Created sales_org feature")
        except Exception as e:
            logger.error(f"Error creating sales_org feature: {str(e)}")
    else:
        logger.warning(f"No sales organization column found. Tried: {sales_org_cols}")
    
    # Log enhanced features
    new_features = set(df.columns) - set(df.columns)
    logger.info(f"Added {len(new_features)} new features: {new_features}")
    
    return df