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

def calculate_forecast_accuracy_depr(actual, forecast, level='overall'):
    """
    Calculate forecast accuracy metrics
    
    Args:
        actual: Actual values (pandas Series or DataFrame)
        forecast: Forecast values (pandas Series or DataFrame)
        level: Level of aggregation ('overall', 'monthly', 'sku', 'region')
        
    Returns:
        DataFrame with accuracy metrics
    """
    if level == 'overall':
        # Overall accuracy
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        accuracy = 100 - mape
        return pd.DataFrame({'MAPE': [mape], 'Accuracy_Pct': [accuracy]})
    
    elif level == 'monthly':
        # Monthly accuracy
        monthly_actual = actual.resample('ME').sum()
        monthly_forecast = forecast.resample('ME').sum()
        
        monthly_accuracy = pd.DataFrame({
            'Month': monthly_actual.index,
            'Actual': monthly_actual.values,
            'Forecast': monthly_forecast.values
        })
        
        monthly_accuracy['Abs_Error'] = np.abs(monthly_accuracy['Actual'] - monthly_accuracy['Forecast'])
        monthly_accuracy['MAPE'] = (monthly_accuracy['Abs_Error'] / monthly_accuracy['Actual']) * 100
        monthly_accuracy['Accuracy_Pct'] = 100 - monthly_accuracy['MAPE']
        
        return monthly_accuracy
    
    elif level == 'sku':
        # Implement SKU level accuracy
        # This would require actual and forecast to be DataFrames with SKU columns
        pass
    
    elif level == 'region':
        # Implement region level accuracy
        # This would require actual and forecast to be DataFrames with region information
        pass
    
    return None
