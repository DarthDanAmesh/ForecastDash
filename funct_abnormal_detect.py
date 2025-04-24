import plotly.express as px
import pandas as pd


def detect_sales_anomalies(ts, window=3, sigma=2):
    """Detect anomalies in sales time series"""
    # Use smaller window for limited data
    rolling_mean = ts.rolling(window=window, min_periods=1).mean()
    rolling_std = ts.rolling(window=window, min_periods=1).std().fillna(ts.std())
    
    upper_bound = rolling_mean + (rolling_std * sigma)
    lower_bound = rolling_mean - (rolling_std * sigma)
    
    anomalies = pd.DataFrame(index=ts.index)
    anomalies['original'] = ts
    anomalies['rolling_mean'] = rolling_mean
    anomalies['upper_bound'] = upper_bound
    anomalies['lower_bound'] = lower_bound
    anomalies['is_anomaly'] = (ts > upper_bound) | (ts < lower_bound)
    
    return anomalies

def detect_sales_anomalies_depr(ts, window=12, sigma=3):
    """
    Detect anomalies in sales time series using rolling statistics
    
    Args:
        ts: Time series data (pandas Series)
        window: Rolling window size in periods
        sigma: Number of standard deviations to consider as anomaly
        
    Returns:
        DataFrame with original data and anomaly flags
    """
    # Create rolling statistics
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    
    # Define upper and lower bounds
    upper_bound = rolling_mean + (rolling_std * sigma)
    lower_bound = rolling_mean - (rolling_std * sigma)
    
    # Identify anomalies
    anomalies = pd.DataFrame(index=ts.index)
    anomalies['original'] = ts
    anomalies['rolling_mean'] = rolling_mean
    anomalies['upper_bound'] = upper_bound
    anomalies['lower_bound'] = lower_bound
    anomalies['is_anomaly'] = (ts > upper_bound) | (ts < lower_bound)
    
    return anomalies

def plot_anomalies(anomalies):
    """Plot time series with anomalies highlighted"""
    fig = px.line(anomalies['original'], title="Sales Anomalies Detection")
    fig.add_scatter(x=anomalies.index, y=anomalies['rolling_mean'], name='Rolling Mean')
    fig.add_scatter(x=anomalies.index, y=anomalies['upper_bound'], 
                   name='Upper Bound', line=dict(dash='dash'))
    fig.add_scatter(x=anomalies.index, y=anomalies['lower_bound'], 
                   name='Lower Bound', line=dict(dash='dash'))
    
    # Highlight anomalies in red
    anomaly_points = anomalies[anomalies['is_anomaly']]
    fig.add_scatter(x=anomaly_points.index, y=anomaly_points['original'], 
                   mode='markers', name='Anomalies', marker=dict(color='red', size=10))
    
    return fig