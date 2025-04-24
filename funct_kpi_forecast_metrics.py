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


def plot_forecast_accuracy_depreee(accuracy_df):
    """Enhanced visualization with anomaly highlighting"""
    fig = px.line(accuracy_df, x='Month', y='Accuracy_Pct',
                 title="Forecast Accuracy with Anomaly Detection",
                 labels={'Accuracy_Pct': 'Accuracy %'})
    
    # Add error bars
    fig.add_trace(go.Bar(x=accuracy_df['Month'],
                      y=accuracy_df['Absolute_Pct_Error'],
                      name='Error %',
                      marker_color='lightcoral'))
    
    # Highlight anomalies
    anomalies = accuracy_df[accuracy_df['Is_Anomaly']]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['Month'],
            y=anomalies['Accuracy_Pct'],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomaly'
        ))
    
    # Add threshold line
    mean_acc = accuracy_df['Accuracy_Pct'].mean()
    fig.add_hline(y=mean_acc, line_dash="dot",
                 annotation_text=f"Mean Accuracy: {mean_acc:.1f}%",
                 annotation_position="bottom right")
    
    fig.update_layout(barmode='overlay', hovermode="x unified")
    return fig