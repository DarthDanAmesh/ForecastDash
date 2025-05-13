import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from cls_data_preprocessor import DataProcessor
from column_config import STANDARD_COLUMNS
from typing import Optional

def plot_historical_forecast_comparison(ts: pd.Series, forecast: pd.Series):
    """Plot bar chart comparing historical demand and forecast."""
    st.subheader("Historical vs. Forecast Demand")
    
    # Prepare historical data for last 3 years
    current_year = ts.index.max().year
    historical_data = {}
    for year_offset in range(3, 0, -1):
        year = current_year - year_offset
        year_data = ts[ts.index.year == year].resample('ME').sum()
        if not year_data.empty:
            historical_data[f"Prior Yr. {year_offset}"] = year_data
    
    # Prepare forecast data
    forecast_df = forecast.to_frame(name='Adjusted Forecast')
    
    # Create figure
    fig = go.Figure()
    
    # Add historical traces
    colors = {'Prior Yr. 3': 'green', 'Prior Yr. 2': 'gray', 'Prior Yr. 1': 'blue'}
    for label, data in historical_data.items():
        fig.add_trace(go.Bar(
            x=data.index,
            y=data.values,
            name=label,
            marker_color=colors[label],
            opacity=0.6
        ))
    
    # Add forecast trace
    fig.add_trace(go.Bar(
        x=forecast_df.index,
        y=forecast_df['Adjusted Forecast'],
        name='Adjusted Forecast',
        marker_color='red',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Demand Comparison",
        xaxis_title="Month",
        yaxis_title="Demand",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="tooltip">ℹ️<span class="tooltiptext">Compare past demand with the forecast to plan inventory.</span></div>',
        unsafe_allow_html=True
    )

def display_kpis():
    """Display KPIs: Safety Stock, MTD Sales, % Achievement."""
    st.subheader("Key Performance Indicators")
    data = st.session_state.state.data
    ts = DataProcessor.prepare_time_series(data)
    if ts is None or ts.empty:
        return
    
    current_month = ts.index.max()
    mtd_sales = ts[ts.index.month == current_month.month].sum()
    avg_demand = ts.mean()
    safety_stock = avg_demand * 0.2  # Example: 20% of average demand
    achievement = (mtd_sales / avg_demand * 100) if avg_demand > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Safety Stock",
            f"{safety_stock:.2f}",
            help="Recommended stock buffer based on average demand."
        )
    with col2:
        st.metric(
            "MTD Sales",
            f"{mtd_sales:.2f}",
            help="Month-to-date sales for the current period."
        )
    with col3:
        st.metric(
            "% Achievement",
            f"{achievement:.2f}%",
            delta=f"{achievement - 100:.2f}%",
            delta_color="inverse",
            help="Sales performance relative to average demand."
        )