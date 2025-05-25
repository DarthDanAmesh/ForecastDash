# funct_shw_forecast_plot.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from cls_data_preprocessor import DataProcessor
from cls_session_management import SessionState
from constants import STANDARD_COLUMNS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def display_forecast(data: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Display SKU-level forecast plot and data."""
    st.subheader("Demand Forecast")
    
    if not isinstance(forecast, pd.DataFrame):
        st.error("Invalid forecast data: Expected a DataFrame.", icon="🚨")
        return
    
    # Check for required columns
    material_col = STANDARD_COLUMNS['material']
    required_cols = ['date', 'forecast', material_col]
    missing_cols = [col for col in required_cols if col not in forecast.columns]
    if missing_cols:
        st.warning(f"Missing columns in forecast: {', '.join(missing_cols)}. Displaying aggregate forecast.", icon="⚠️")
        materials = ['All']
        material_historical = data.groupby([STANDARD_COLUMNS['date']])[STANDARD_COLUMNS['demand']].sum().reset_index()
        material_forecast = forecast
    else:
        # Aggregate historical data by material
        historical = data.groupby([STANDARD_COLUMNS['date'], material_col])[STANDARD_COLUMNS['demand']].sum().reset_index()
        materials = forecast[material_col].unique()
    
    fig = go.Figure()
    
    # Determine the date range for the x-axis
    historical_dates = pd.to_datetime(historical[STANDARD_COLUMNS['date']])
    forecast_dates = pd.to_datetime(forecast['date'])
    min_date = min(historical_dates.min(), forecast_dates.min())
    max_date = forecast_dates.max()  # Forecast should extend 6 months from end of historical data
    
    for material in materials:
        if material == 'All':
            material_historical = historical
            material_forecast = forecast
        else:
            material_historical = historical[historical[material_col] == material]
            material_forecast = forecast[forecast[material_col] == material]
        
        # Historical trace
        fig.add_trace(go.Scatter(
            x=material_historical[STANDARD_COLUMNS['date']],
            y=material_historical[STANDARD_COLUMNS['demand']],
            name=f'{material} Historical',
            mode='lines',
            line=dict(width=2)
        ))
        
        # Forecast trace
        fig.add_trace(go.Scatter(
            x=material_forecast['date'],
            y=material_forecast['forecast'],
            name=f'{material} Forecast',
            mode='lines',
            line=dict(width=2, dash='dash')
        ))
        
        # Confidence intervals
        if 'lower_bound' in material_forecast.columns and 'upper_bound' in material_forecast.columns:
            fig.add_trace(go.Scatter(
                x=material_forecast['date'].tolist() + material_forecast['date'].tolist()[::-1],
                y=material_forecast['upper_bound'].tolist() + material_forecast['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{material} Confidence Interval',
                showlegend=True
            ))

    fig.update_layout(
        title="Demand Forecast by SKU",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=30, l=30, r=30),
        xaxis=dict(
            range=[min_date, max_date + pd.offsets.MonthEnd(1)]  # Extend slightly beyond max forecast date
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Forecast Values:")
    display_cols = ['date', 'forecast']
    if material_col in forecast.columns:
        display_cols.insert(0, material_col)
    if 'lower_bound' in forecast.columns and 'upper_bound' in forecast.columns:
        display_cols.extend(['lower_bound', 'upper_bound'])
    st.dataframe(
        forecast[display_cols],
        use_container_width=True
    )
    st.download_button(
        label="Download Forecast",
        data=forecast.to_csv(index=False),
        file_name="forecast.csv",
        mime="text/csv",
        help="Download the forecast data as CSV."
    )

def show_forecasting():
    """Render forecasting UI based on mode."""
    if not isinstance(st.session_state.state, SessionState):
        st.error("Session state not initialized. Reinitializing...", icon="🚨")
        st.session_state.state = SessionState.get_or_create()
        st.rerun()

    if st.session_state.state.data is None:
        st.error("No data loaded. Please upload a CSV file.", icon="🚨")
        return

    data = DataProcessor.preprocess_data(st.session_state.state.data)
    if data.empty:
        st.error("Failed to preprocess data. Ensure valid date, demand, and material columns.", icon="🚨")
        return

    if st.session_state.mode == "Simple":
        show_simple_forecast(data)
    else:
        show_technical_forecast(data)

def show_simple_forecast(data: pd.DataFrame):
    """Simplified forecasting UI for non-technical users."""
    st.header("Demand Forecast")
    st.markdown("View predicted demand by SKU.")
    
    if "DeepAR" in st.session_state.state.forecasts:
        display_forecast(data, st.session_state.state.forecasts["DeepAR"])
    else:
        st.info("No forecast available. Generate a forecast in the main interface.", icon="ℹ️")

def show_technical_forecast(data: pd.DataFrame):
    """Detailed forecasting UI for technical users."""
    st.header("Demand Forecasting")
    st.markdown("View SKU-level forecasts.")
    
    if "DeepAR" in st.session_state.state.forecasts:
        display_forecast(data, st.session_state.state.forecasts["DeepAR"])
    else:
        st.info("No forecast available. Generate a forecast in the main interface.", icon="ℹ️")