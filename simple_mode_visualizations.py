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
    """Display key performance indicators for demand planners."""
    st.subheader("Key Performance Indicators")

    data = st.session_state.state.data
    ts = DataProcessor.prepare_time_series(data)

    if ts is None or ts.empty:
        st.warning("No valid time series data available for KPIs.")
        return

    # Ensure index is datetime
    ts = ts.dropna()
    ts.index = pd.to_datetime(ts.index)
    current_date = ts.index.max()

    # Define current and previous month/year
    current_month = current_date.to_period('M')
    prev_month = (current_date - pd.DateOffset(months=1)).to_period('M')

    # Filter full months only for MoM comparison
    def filter_month(df, period):
        return df[df.index.to_period('M') == period]

    current_month_data = filter_month(ts, current_month)
    prev_month_data = filter_month(ts, prev_month)

    mtd_sales = current_month_data.sum()
    prev_month_sales = prev_month_data.sum() if not prev_month_data.empty else 0

    # Average Demand (3-month rolling average for smoother metric)
    avg_demand = ts[-90:].mean()  # Approximate last 3 months

    # Safety Stock based on volatility
    demand_volatility = (ts[-90:].std() / avg_demand * 100) if avg_demand > 0 else 0
    safety_stock = avg_demand * (1 + (demand_volatility / 100) * 0.5)  # Adjust buffer based on volatility

    # % Achievement
    achievement = (mtd_sales / avg_demand * 100) if avg_demand > 0 else 0
    achievement_delta = achievement - 100  # vs target

    # Forecasted Demand (simple moving average)
    forecasted_demand = ts.rolling(window=3, min_periods=1).mean().iloc[-1]

    # MoM Change (only compare full months to avoid partial month bias)
    mom_change = 0
    if not prev_month_data.empty and prev_month_sales > 0:
        mom_change = ((mtd_sales - prev_month_sales) / prev_month_sales * 100)

    # Layout
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Safety Stock", f"{safety_stock:.2f}",
                  help="Calculated using average demand and volatility. Buffer increases with higher variability.")

    with kpi2:
        st.metric("MTD Sales", f"{mtd_sales:.2f}",
                  help="Total sales from the current month-to-date.")

    with kpi3:
        st.metric("% Achievement", f"{achievement:.1f}%",
                  delta=f"{achievement_delta:.1f}%",
                  delta_color="normal" if achievement_delta >= 0 else "inverse",
                  help="Current MTD sales as % of recent average demand.")

    kpi4, kpi5, kpi6 = st.columns(3)
    with kpi4:
        st.metric("Forecasted Demand", f"{forecasted_demand:.2f}",
                  help="Next expected period demand using 3-period moving average.")

    with kpi5:
        st.metric("Demand Volatility", f"{demand_volatility:.1f}%",
                  help="Coefficient of variation (standard deviation / mean), indicates demand stability.")

    with kpi6:
        st.metric("MoM Sales Change", f"{mom_change:.1f}%",
                  delta=f"{mom_change:.1f}%",
                  delta_color="normal" if mom_change >= 0 else "inverse",
                  help="Change in total monthly sales compared to last full month.")
