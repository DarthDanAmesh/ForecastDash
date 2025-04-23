import streamlit as st
from cls_data_preprocessor import DataProcessor
from cls_forecast_engine import ForecastEngine
from cls_plots_visuals import Visualizer
from consts_model import DEFAULT_FORECAST_PERIOD

def show_forecasting():
    st.header("Demand Forecasting")
    
    if not st.session_state.state.models:
        st.warning("No trained models available. Please train a model first.")
        return
    
    model_type = st.selectbox("Select Model for Forecasting", 
                             list(st.session_state.state.models.keys()))
    
    model_info = st.session_state.state.models[model_type]
    forecast_period = st.number_input(
                                        "Forecast Period (months)", 
                                        min_value=1, max_value=24, 
                                        value=DEFAULT_FORECAST_PERIOD,
                                        key="forecast_period_input"
                                    )
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            df = st.session_state.state.data
            ts = DataProcessor.prepare_time_series(df)
            
            if model_type == "XGBoost":
                last_date = DataProcessor.prepare_time_series(df).index[-1]
                forecast = ForecastEngine.forecast_xgboost(
                    model_info['model'],
                    model_info['last_values'],
                    forecast_period,
                    last_date
                )

            elif model_type == "ARIMA":
                forecast = ForecastEngine.forecast_arima(
                    model_info['model'], 
                    forecast_period
                )
            
            st.session_state.state.forecasts[model_type] = forecast
            
            # Show forecast
            st.subheader("Demand Forecast")
            st.plotly_chart(Visualizer.plot_forecast(ts, forecast))
            
            # Show forecast values
            st.write("Forecast Values:")
            st.dataframe(forecast.to_frame(name='Forecast'))