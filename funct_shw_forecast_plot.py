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
    
    # Aggregate historical data by material
    historical = data.groupby([STANDARD_COLUMNS['date'], STANDARD_COLUMNS['material']])[STANDARD_COLUMNS['demand']].sum().reset_index()
    
    # Create subplot for each material
    materials = forecast[STANDARD_COLUMNS['material']].unique()
    fig = go.Figure()
    
    for material in materials:
        material_historical = historical[historical[STANDARD_COLUMNS['material']] == material]
        material_forecast = forecast[forecast[STANDARD_COLUMNS['material']] == material]
        
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
        margin=dict(t=70, b=30, l=30, r=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Forecast Values:")
    st.dataframe(
        forecast[[STANDARD_COLUMNS['material'], 'date', 'forecast', 'lower_bound', 'upper_bound']],
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