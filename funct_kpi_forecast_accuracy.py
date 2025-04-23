import pandas as pd
import numpy as np
import plotly.express as px

def calculate_forecast_accuracy(actual, forecast, level='overall'):
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

def plot_forecast_accuracy(accuracy_df, level='monthly'):
    """Create visualization for forecast accuracy"""
    if level == 'monthly':
        fig = px.line(accuracy_df, x='Month', y='Accuracy_Pct', 
                     title="Forecast Accuracy by Month (%)")
        fig.add_bar(x=accuracy_df['Month'], y=accuracy_df['MAPE'], name='Error %')
        fig.update_layout(yaxis_title='Percentage', xaxis_title='Month')
        return fig
    
    # Add other visualization types for different levels
    return None