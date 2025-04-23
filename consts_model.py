import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Constants
MODEL_TYPES = {
    "XGBoost": xgb.XGBRegressor,
    "ARIMA": ARIMA
}
DEFAULT_FORECAST_PERIOD = 6  # months