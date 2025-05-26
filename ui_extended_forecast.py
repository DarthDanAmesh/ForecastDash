import streamlit as st
import pandas as pd
import plotly.express as px

from cls_data_preprocessor import DataProcessor
from forecast_engine import ForecastEngine

from funct_kpi_forecast_metrics import calculate_forecast_accuracy 
from funct_prep_order_based_deliv_forecast import prepare_order_time_series


def show_extended_forecasting():
    st.header("Extended Demand Forecasting (12 Months)")
    
    if not st.session_state.state.models:
        st.warning("No trained models available. Please train a model first.")
        return
    
    model_type = st.selectbox("Select Model for Extended Forecasting", 
                             list(st.session_state.state.models.keys()),
                             key="extended_forecast_model")
    
    model_info = st.session_state.state.models[model_type]
    
    # Fixed 12-month forecast period as per business requirement
    forecast_period = 12
    st.info(f"Forecasting for the next {forecast_period} months as per business requirement.")
    
    # Add confidence interval option
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)
    confidence_level = st.slider("Confidence Level (%)", 50, 95, 80) if show_confidence else 80
    
    if st.button("Generate Delivery Forecast"):
        with st.spinner("Generating extended forecast..."):
            df = st.session_state.state.data
            
            # Use order-based time series instead of delivery-based
            ts = prepare_order_time_series(df)
            if ts is None:
                ts = DataProcessor.prepare_time_series(df)  # Fallback to delivery-based
                st.info("Using delivery based time series (Act. Gds Issue Date)")
            
            # Initialize forecast variable
            forecast = None
            lower_bound = None
            upper_bound = None

            if model_type == "XGBoost":
                last_date = DataProcessor.prepare_time_series(df).index[-1]
                forecast = ForecastEngine.forecast_xgboost(
                    model_info['model'], 
                    model_info['last_values'], 
                    forecast_period,
                    last_date
                )
                
                # Generate simple confidence intervals
                forecast_series = pd.Series(forecast) if not isinstance(forecast, pd.Series) else forecast
                
                error_margin = (1 - confidence_level/100) * 0.5

                # Calculate bounds with minimum 10% spread even for high confidence
                lower_bound = forecast_series * (1 - max(error_margin, 0.1))
                upper_bound = forecast_series * (1 + max(error_margin, 0.1))
                
            elif model_type == "ARIMA":
                forecast_result = model_info['model'].get_forecast(steps=forecast_period)
                forecast = forecast_result.predicted_mean
                
                # Manual confidence intervals if not available from model
                error_margin = (1 - confidence_level/100) * 0.5
                lower_bound = forecast * (1 - max(error_margin, 0.1))
                upper_bound = forecast * (1 + max(error_margin, 0.1))
            
            # Add a check to ensure forecast was generated
            if forecast is None:
                st.error(f"Forecast generation failed or model type '{model_type}' is not supported for extended forecasting.", icon="🚨")
                return # Exit the function if forecast is not generated

            st.session_state.state.forecasts[model_type] = forecast

            # Show forecast
            st.subheader("12-Month Demand Forecast")

            # Create forecast plot with confidence intervals
            fig = px.line(ts, title="Extended Demand Forecast (12 Months)")
            fig.add_scatter(x=forecast.index, y=forecast.values, name='Forecast')

            if show_confidence and lower_bound is not None and upper_bound is not None:
                fig.add_scatter(x=forecast.index,
                                y=upper_bound,
                               name=f'Upper Bound ({confidence_level}%)',
                               line=dict(dash='dash', color='lightgreen'))
                fig.add_scatter(x=forecast.index, y=lower_bound,
                               name=f'Lower Bound ({confidence_level}%)',
                               line=dict(dash='dash', color='lightcoral'),
                               fill='tonexty',
                               fillcolor='rgba(255, 204, 204, 0.2)'
                               )

            st.plotly_chart(fig)

            # Show forecast values in a table
            forecast_df = pd.DataFrame({
                'Date': forecast.index,
                'Forecast': forecast.values
            })

            if show_confidence and lower_bound is not None and upper_bound is not None:
                # Ensure these have the same length as forecast
                forecast_df['Lower_Bound'] = lower_bound.values if hasattr(lower_bound, 'values') else lower_bound
                forecast_df['Upper_Bound'] = upper_bound.values if hasattr(upper_bound, 'values') else upper_bound


            st.write("Extended Forecast Values:")
            st.dataframe(forecast_df)