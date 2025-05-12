# column_config.py
"""
Centralized column configuration for the demand planning pipeline
"""

STANDARD_COLUMNS = {
    'date': 'date',
    'demand': 'demand',
    'country': 'Country Key Ship-to',
    'material': 'Material',
    'order_date': 'Customer Ref. Date',
    'delivery_date': 'Act. Gds Issue Date',
    'planned_delivery_date': 'Pland Gds Mvmnt Date',
    'delivery_quantity': 'Delivery Quantity',
    'act_gds_issue_date': 'Act. Gds Issue Date',
    'customer_reference': 'Customer Reference'
}

COLUMN_ALIASES = {
    'date': [
        'date', 'created on', 'created_on', 'timestamp', 'order date', 'document date', 'customer ref. date'
    ],
    'demand': [
        'quantity', 'delivery quantity', 'sales', 'orders', 'value', 'volume', 'deliv.value lips doc.curr.', 'qty'
    ],
    'country': [
        'country', 'country_code', 'ship_to_country', 'customer country', 'region', 'country key ship-to'
    ],
    'material': [
        'material', 'product', 'item', 'sku', 'article', 'customer material'
    ],
    'order_date': [
        'order date', 'customer ref. date', 'document date', 'order_date'
    ],
    'delivery_date': [
        'delivery date', 'act. gds issue date', 'shipment date', 'ship_date', 'act_gds_issue_date'
    ],
    'planned_delivery_date': [
        'planned delivery date', 'pland gds mvmnt date', 'expected delivery date'
    ],
    'delivery_quantity': [
        'delivery quantity', 'quantity', 'qty', 'deliv qty', 'delivery', 'delivery_quantity'
    ],
    'act_gds_issue_date': [
        'act. gds issue date', 'actual goods issue', 'shipment date', 'act_gds_issue_date'
    ],
    'customer_reference': [
        'customer reference', 'cust ref', 'reference'
    ]
}


def standardize_column_names(df):
    """
    Standardize column names to internal standard keys (e.g., 'demand', 'country').
    """
    df.columns = [str(col).strip().lower() for col in df.columns]
    column_mapping = {}

    for std_key, aliases in COLUMN_ALIASES.items():
        for alias in [a.lower() for a in aliases]:
            if alias in df.columns:
                column_mapping[alias] = std_key  # Rename to 'demand', 'country', etc.
                break

    return df.rename(columns=column_mapping)



def get_standard_name(original_name):
    """
    Get the standard name for a given original column name.
    """
    original_name = str(original_name).strip().lower()
    for std_name, aliases in COLUMN_ALIASES.items():
        if original_name in [a.lower() for a in aliases]:
            return std_name
    return original_name