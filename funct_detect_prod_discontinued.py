import numpy as np

def detect_discontinued_products(df, threshold_months=3):
    """
    Detect potentially discontinued products based on ordering patterns
    """
    # Group by product and get the last order date
    product_last_order = df.groupby('Material')['Created On'].max().reset_index()
    product_last_order.columns = ['Material', 'Last_Order_Date']
    
    # Get current date from the data
    latest_date = df['Created On'].max()
    
    # Calculate days since last order
    product_last_order['Days_Since_Last_Order'] = (latest_date - product_last_order['Last_Order_Date']).dt.days
    
    # Convert to months (approximate)
    product_last_order['Months_Since_Last_Order'] = (product_last_order['Days_Since_Last_Order'] / 30).round(1)
    
    # Flag potentially discontinued products
    product_last_order['Potentially_Discontinued'] = product_last_order['Months_Since_Last_Order'] > threshold_months
    
    # Add product description
    product_info = df[['Material', 'Material Description']].drop_duplicates()
    discontinued_products = product_last_order.merge(product_info, on='Material')
    
    return discontinued_products.sort_values('Months_Since_Last_Order', ascending=False)


def detect_discontinued_products_Depr(df, threshold_months=3):
    """
    Detect potentially discontinued products based on ordering patterns
    
    Args:
        df: DataFrame with order data
        threshold_months: Number of months without orders to flag as potentially discontinued
        
    Returns:
        DataFrame with discontinued product flags
    """
    # Group by product and get the last order date
    product_last_order = df.groupby('Material')['Created On'].max().reset_index()
    product_last_order.columns = ['Material', 'Last_Order_Date']
    
    # Get current date from the data
    latest_date = df['Created On'].max()
    
    # Calculate months since last order
    product_last_order['Months_Since_Last_Order'] = ((latest_date - product_last_order['Last_Order_Date']) 
                                                   / np.timedelta64(1, 'M')).round(1)
    
    # Flag potentially discontinued products
    product_last_order['Potentially_Discontinued'] = product_last_order['Months_Since_Last_Order'] > threshold_months
    
    # Add product description
    product_info = df[['Material', 'Material Description']].drop_duplicates()
    discontinued_products = product_last_order.merge(product_info, on='Material')
    
    return discontinued_products.sort_values('Months_Since_Last_Order', ascending=False)