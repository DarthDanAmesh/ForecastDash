# constants.py
from typing import Dict, List

# Standardized column names
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

# Column aliases for flexible input
COLUMN_ALIASES = {
    'date': ['date', 'created on', 'Created On', 'customer ref. date', 'Customer Ref. Date'],
    'demand': ['deliv.value lips doc.curr.', 'value', 'Delivery Quantity', 'delivery quantity', 'quantity'],
    'material': ['material', 'Material', 'customer material', 'Customer Material', 'itemcode'],
    'country': ['country', 'Country Key Ship-to', 'country key ship-to'],
    'delivery_date': ['act. gds issue date', 'Act. Gds Issue Date'],
    'planned_delivery_date': ['pland gds mvmnt date', 'Pland Gds Mvmnt Date'],
    'order_date': ['order date', 'Customer Ref. Date'],
    'delivery_quantity': ['delivery quantity', 'Delivery Quantity'],
}

# Global settings
DEFAULT_FREQ = 'ME'  # Weekly frequency, aligned with test.py
DEFAULT_PREDICTION_LENGTH = 6  # Number of forecast steps, aligned with test.py
MIN_OBSERVATIONS_PER_MATERIAL = 8  # Minimum data points per material
MAX_ENCODER_LENGTH = 6  # DeepAR context window
BATCH_SIZE = 32  # Training batch size
MAX_EPOCHS = 10  # Maximum training epochs

DEFAULT_FORECAST_PERIOD = 6  # months

# Constants (defined at module level)
COUNTRY_CODE_MAP = {
    'IT': 'Italy',
    'DE': 'Germany',
    'FR': 'France',
    'ES': 'Spain',
    'GB': 'United Kingdom',
    'NL': 'Netherlands',
    'BE': 'Belgium',
    'PT': 'Portugal',
    'CH': 'Switzerland',
    'AT': 'Austria',
    'SE': 'Sweden',
    'NO': 'Norway',
    'FI': 'Finland',
    'DK': 'Denmark',
    'IE': 'Ireland',
    'PL': 'Poland',
    'CZ': 'Czech Republic',
    'HU': 'Hungary',
    'RO': 'Romania',
    'GR': 'Greece',
    'TR': 'Turkey',
}
