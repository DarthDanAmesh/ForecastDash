from constants import STANDARD_COLUMNS

def prepare_order_time_series(df, target_col=STANDARD_COLUMNS['delivery_quantity'], 
                             date_col=STANDARD_COLUMNS['order_date'], freq='ME'):
    """
    Convert dataframe to order-based time series format instead of delivery-based
    
    Args:
        df: DataFrame with order data
        target_col: Column name for the quantity to forecast
        date_col: Column name for the order date
        freq: Frequency for resampling ('D', 'W', 'ME', etc.)
        
    Returns:
        Time series of orders
    """
    if date_col not in df.columns:
        return None
        
    df_orders = df.dropna(subset=[date_col])
    
    ts_orders = df_orders.set_index(date_col)[target_col]
    ts_orders = ts_orders.resample(freq).sum()
    
    return ts_orders