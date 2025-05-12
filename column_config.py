"""
Centralized column configuration for the demand planning pipeline
"""

# Standardized column names (our internal naming convention)
STANDARD_COLUMNS = {
    'date': 'date',
    'demand': 'demand',
    'country': 'Country Key Ship-to',
    'material': 'Material',
    'order_date': 'Customer Ref. Date',
    'delivery_date': 'Act. Gds Issue Date',
    'planned_delivery_date': 'Pland Gds Mvmnt Date',
    'delivery_quantity': 'Delivery Quantity',
    'act_gds_issue_date': 'Act. Gds Issue Date'
}

# Common variations of column names we might encounter in input data
COLUMN_ALIASES = {
    'date': ['date', 'created on', 'created_on', 'timestamp', 'order date', 'document date'],
    'demand': ['quantity', 'delivery quantity', 'sales', 'orders', 'value', 'volume'],
    'country': ['country', 'country_code', 'ship_to_country', 'customer country', 'region'],
    'material': ['material', 'product', 'item', 'sku', 'article', 'Material'],
    'order_date': ['order date', 'customer ref. date', 'document date', 'order_date'],
    'delivery_date': ['delivery date', 'act. gds issue date', 'shipment date', 'ship_date'],
    'planned_delivery_date': ['planned delivery date', 'pland gds mvmnt date', 'expected delivery date'],
    'delivery_quantity': ['delivery quantity', 'quantity', 'qty', 'deliv qty', 'Delivery Quantity'],
    'act_gds_issue_date': ['act. gds issue date', 'actual goods issue', 'shipment date', 'Act. Gds Issue Date']
}

def standardize_column_names(df):
    """
    Standardize column names in a DataFrame based on known aliases
    
    Args:
        df (pd.DataFrame): Input DataFrame with arbitrary column names
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    # Convert all columns to lowercase for matching
    df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
    
    # Create mapping from current column names to standard names
    column_mapping = {}
    for std_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                column_mapping[alias] = std_name
    
    # Rename columns to standardized names
    df = df.rename(columns=column_mapping)
    
    return df

def get_standard_name(original_name):
    """
    Get the standard name for a given original column name
    
    Args:
        original_name (str): Original column name
        
    Returns:
        str: Standardized column name or original name if not found
    """
    # Convert to lowercase for matching
    original_name = original_name.lower()
    
    # Check each set of aliases
    for std_name, aliases in COLUMN_ALIASES.items():
        if original_name in aliases:
            return std_name
            
    return original_name