import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Constants
MODEL_TYPES = {
    "XGBoost": xgb.XGBRegressor,
    "ARIMA": ARIMA
}
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
