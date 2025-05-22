# funct_kpi_forecast_metrics.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def calculate_forecast_accuracy(actual, forecast, level='overall'):
    """Calculate forecast accuracy metrics"""
    # Handle NaN values from shift operation
    valid_indices = ~(np.isnan(actual) | np.isnan(forecast))
    actual_valid = actual[valid_indices]
    forecast_valid = forecast[valid_indices]
    if len(actual_valid) == 0:
        return pd.DataFrame({'MAPE': [100], 'Accuracy_Pct': [0]})
    
    if level == 'overall':
        # Avoid division by zero
        actual_nonzero = np.where(actual_valid == 0, 0.0001, actual_valid)
        mape = np.mean(np.abs((actual_valid - forecast_valid) / actual_nonzero)) * 100
        mape = min(mape, 100)  # Cap at 100%
        accuracy = 100 - mape
        return pd.DataFrame({'MAPE': [mape], 'Accuracy_Pct': [accuracy]})
    
    elif level == 'monthly':
        # For monthly data that's already aggregated
        result = pd.DataFrame({
            'Month': actual.index,
            'Actual': actual.values,
            'Forecast': forecast.values
        })
        # Handle zeros and NaNs
        result = result.dropna()
        result['Actual_Safe'] = np.where(result['Actual'] == 0, 0.0001, result['Actual'])
        result['Abs_Error'] = np.abs(result['Actual'] - result['Forecast'])
        result['MAPE'] = (result['Abs_Error'] / result['Actual_Safe']) * 100
        result['MAPE'] = result['MAPE'].clip(0, 100)  # Cap at 100%
        result['Accuracy_Pct'] = 100 - result['MAPE']
        return result

