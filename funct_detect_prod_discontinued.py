# funct_detect_prod_discontinued.py
import pandas as pd
from constants import STANDARD_COLUMNS

def detect_discontinued_products(df: pd.DataFrame, threshold_months: int = 3, forecast: pd.DataFrame = None) -> pd.DataFrame:
    """Detect potentially discontinued SKUs based on order patterns and forecasts."""
    if not all(col in df.columns for col in [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['material']]):
        raise ValueError("DataFrame must include date and material columns")
    
    # Last order date per SKU
    product_last_order = df.groupby(STANDARD_COLUMNS['material'])[STANDARD_COLUMNS['date']].max().reset_index()
    product_last_order.columns = [STANDARD_COLUMNS['material'], 'Last_Order_Date']
    
    # Calculate months since last order
    latest_date = df[STANDARD_COLUMNS['date']].max()
    product_last_order['Months_Since_Last_Order'] = ((latest_date - product_last_order['Last_Order_Date']).dt.days / 30).round(1)
    product_last_order['Potentially_Discontinued'] = product_last_order['Months_Since_Last_Order'] > threshold_months
    
    # Forecast insights
    if forecast is not None and isinstance(forecast, pd.DataFrame) and not forecast.empty:
        forecast_summary = forecast.groupby(STANDARD_COLUMNS['material'])['forecast'].mean().reset_index()
        forecast_summary.columns = [STANDARD_COLUMNS['material'], 'Average_Forecast_Demand']
        product_last_order = product_last_order.merge(forecast_summary, on=STANDARD_COLUMNS['material'], how='left')
    
    return product_last_order.sort_values('Months_Since_Last_Order', ascending=False)