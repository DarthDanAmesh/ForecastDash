import streamlit as st
from cls_data_preprocessor import DataProcessor
from cls_forecast_engine import ForecastEngine
from cls_plots_visuals import Visualizer
from consts_model import DEFAULT_FORECAST_PERIOD
from funct_kpi_forecast_metrics import calculate_forecast_accuracy

import traceback
import calendar
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# Add to constants.py
MIN_DATA_POINTS = 3
ANOMALY_THRESHOLD = 50  # %
WARNING_THRESHOLD = 5  # % change for trends




def show_forecasting():
    st.header("Demand Forecasting")

    if st.session_state.state.data is None:
        st.error("No data loaded. Please load data first.")
        return

    if len(st.session_state.state.data) < MIN_DATA_POINTS:
        st.error(f"Insufficient data points (minimum {MIN_DATA_POINTS} required)")
        return

    if not st.session_state.state.models:
        st.warning("No trained models available. Please train a model first.")
        return

    # Create tabs for forecast generation and analysis
    tab1, tab2 = st.tabs(["Generate Forecast", "Analyze Accuracy"])
    
    # Initialize session state variables if they don't exist
    if 'current_forecast' not in st.session_state:
        st.session_state.current_forecast = None
    if 'current_ts' not in st.session_state:
        st.session_state.current_ts = None
    if 'forecast_generated' not in st.session_state:
        st.session_state.forecast_generated = False
    if 'forecast_params' not in st.session_state:
        st.session_state.forecast_params = {'model_type': None, 'forecast_period': None}
    
    with tab1:
        model_type = st.selectbox("Select Model for Forecasting", list(st.session_state.state.models.keys()), key="forecast_model_select")
        model_info = st.session_state.state.models[model_type]

        forecast_period = st.number_input("Forecast Period (months)", min_value=1, max_value=24, value=DEFAULT_FORECAST_PERIOD, key="forecast_period_input")

        # Check if parameters have changed from the last forecast
        params_changed = (model_type != st.session_state.forecast_params['model_type'] or 
                         forecast_period != st.session_state.forecast_params['forecast_period'])
        
        if params_changed and st.session_state.forecast_generated:
            st.warning("Forecast parameters changed. Click 'Generate Forecast' to update.")
            st.session_state.forecast_generated = False

        if forecast_period > 12:
            st.warning("Long forecast periods may impact performance. Consider smaller increments.")
            proceed_anyway = st.checkbox("Proceed anyway", key="proceed_checkbox")
            if not proceed_anyway:
                pass  # Don't return, just continue

        # Always display existing forecast if available
        if st.session_state.forecast_generated and st.session_state.current_forecast is not None and not params_changed:
            st.success("Forecast already generated")
            st.subheader("Demand Forecast")
            st.plotly_chart(Visualizer.plot_forecast(st.session_state.current_ts, st.session_state.current_forecast))
            st.write("Forecast Values:")
            st.dataframe(st.session_state.current_forecast.to_frame(name='Forecast'))
        
        if st.button("Generate Forecast", key="generate_forecast_button"):
            with st.spinner("Generating forecast..."):
                try:
                    df = st.session_state.state.data
                    ts = DataProcessor.prepare_time_series(df)
                    # Store time series in session state
                    st.session_state.current_ts = ts
                    
                    if model_type == "XGBoost":
                        last_date = ts.index[-1]
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
                
                    # Update forecast parameters in session state
                    st.session_state.forecast_params = {'model_type': model_type, 'forecast_period': forecast_period}
                    
                    # Store all forecast data in session state
                    st.session_state.state.forecasts[model_type] = {
                        'forecast': forecast,
                        'timestamp': pd.Timestamp.now()
                    }
                    st.session_state.current_forecast = forecast
                    st.session_state.forecast_generated = True

                    # Store for accuracy analysis
                    st.session_state.state.last_forecast = {
                        'type': model_type,
                        'period': forecast_period,
                        'data': forecast
                    }

                    st.success("Forecast generated successfully")
                    st.subheader("Demand Forecast")
                    st.plotly_chart(Visualizer.plot_forecast(ts, forecast))
                    st.write("Forecast Values:")
                    st.dataframe(forecast.to_frame(name='Forecast'))
                    
                except Exception as e:
                    st.error(f"Forecast failed: {str(e)}")
                    st.code(traceback.format_exc())
    
    with tab2:
        if not st.session_state.forecast_generated or st.session_state.current_forecast is None:
            st.warning("No forecast available for analysis. Please generate a forecast first in the 'Generate Forecast' tab.")
            return
        
        st.subheader("Historical Forecast Accuracy Analysis")
        
        # Use data from session state
        if hasattr(st.session_state.state, 'last_forecast'):
            forecast = st.session_state.state.last_forecast
            model_type = forecast['type']
            forecast_period = forecast['period']
        else:
            # Fallback if state object doesn't have last_forecast
            model_type = list(st.session_state.state.models.keys())[0]
            forecast_period = DEFAULT_FORECAST_PERIOD
        
        ts = st.session_state.current_ts
        
        if ts is None or len(ts) < MIN_DATA_POINTS:
            st.warning("Insufficient historical data for accuracy analysis.")
            return
            
        model_info = st.session_state.state.models[model_type]
        time_options = ["Last 3 months", "Last 6 months", "Last 12 months", "All available data"]
        selected_period = st.selectbox("Select Time Period for Analysis", time_options, key="time_period_select")
        months = {"Last 3 months": 3, "Last 6 months": 6, "Last 12 months": 12}.get(selected_period, len(ts))
        months = min(months, len(ts))

        if st.button("Analyze Forecast Accuracy", key="analyze_accuracy_button"):
            with st.spinner("Analyzing forecast accuracy..."):
                try:
                    analysis_data = ts[-months:] if months < len(ts) else ts

                    if len(analysis_data) < MIN_DATA_POINTS + 1:
                        st.warning("Insufficient data points for selected period.")
                        return

                    accuracy_metrics = pd.DataFrame()
                    min_training_size = MIN_DATA_POINTS

                    # Use your calculate_forecast_accuracy function for each time step
                    for i in range(min_training_size, len(analysis_data)):
                        train_data = analysis_data.iloc[:i]
                        test_point = analysis_data.iloc[i]

                        if model_type == "XGBoost":
                            train_data_subset = analysis_data.iloc[:i]
                            df_features = train_data_subset.to_frame()  # Convert to DataFrame
                            target_col = df_features.columns[0]  # Get the column name from DataFrame
                            
                            # Generate features
                            features_df = ForecastEngine.create_features(df_features, target_col)
                            X_features = features_df.drop(columns=[target_col, 'date'])
                            
                            # Safeguard against empty features
                            if X_features.empty:
                                st.error(f"Feature generation failed for i={i}. Skipping this iteration.")
                                continue  # Skip this iteration
                            
                            last_features = X_features.iloc[-1].values  # Now safe
                            last_date = train_data_subset.index[-1]
                            pred = ForecastEngine.forecast_xgboost(
                                model_info['model'],
                                last_features,
                                1,
                                last_date
                            ).iloc[0]
                        elif model_type == "ARIMA":
                            pred = ForecastEngine.forecast_arima(model_info['model'], 1).iloc[0]
                        else:
                            continue

                        actual = test_point
                        
                        # Create mini-series for the accuracy calculation
                        actuals_series = pd.Series([actual], index=[analysis_data.index[i]])
                        forecast_series = pd.Series([pred], index=[analysis_data.index[i]])
                        
                        # Calculate accuracy for this point
                        point_accuracy = calculate_forecast_accuracy(actuals_series, forecast_series, level='monthly')
                        
                        # Add date information
                        point_accuracy['Month'] = analysis_data.index[i]
                        
                        # Check if it's an anomaly
                        is_anomaly = point_accuracy['Accuracy_Pct'].iloc[0] < ANOMALY_THRESHOLD
                        point_accuracy['Is_Anomaly'] = is_anomaly
                        
                        # Append to the metrics dataframe
                        accuracy_metrics = pd.concat([accuracy_metrics, point_accuracy], ignore_index=True)

                    # Error_Pct column added to the df
                    accuracy_metrics['Error_Pct'] = 100 - accuracy_metrics['Accuracy_Pct']

                    # Store metrics for charts and calculations
                    accuracy_numeric = accuracy_metrics['Accuracy_Pct'].copy()
                    error_numeric = accuracy_metrics['Error_Pct'].copy()
                    
                    # Display summary metrics first
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Accuracy", f"{accuracy_numeric.mean():.2f}%")
                    col2.metric("Average Error", f"{error_numeric.mean():.2f}%")
                    col3.metric("Anomalies Detected", f"{accuracy_metrics['Is_Anomaly'].sum()}/{len(accuracy_metrics)}")

                    # Display charts and detailed metrics
                    st.subheader("Monthly Forecast Accuracy")
                    fig = create_enhanced_accuracy_chart(accuracy_metrics)
                    st.plotly_chart(fig)

                    st.subheader("Detailed Accuracy Metrics")
                    # Create a copy for display formatting
                    display_metrics = accuracy_metrics.copy()
                    display_metrics['Month'] = pd.to_datetime(display_metrics['Month']).dt.strftime('%b %Y')
                    display_metrics['Accuracy_Pct'] = display_metrics['Accuracy_Pct'].round(2).astype(str) + '%'
                    display_metrics['MAPE'] = display_metrics['MAPE'].round(2).astype(str) + '%'

                    st.dataframe(
                        display_metrics.style.apply(
                            lambda row: ['background-color: #ffcccc' if row['Is_Anomaly'] else '' for _ in row],
                            axis=1
                        )
                    )

                    # Calculate overall accuracy
                    overall_accuracy = calculate_forecast_accuracy(
                        pd.Series(accuracy_metrics['Actual'].values, index=accuracy_metrics['Month']),
                        pd.Series(accuracy_metrics['Forecast'].values, index=accuracy_metrics['Month'])
                    )
                    
                    st.subheader("Overall Forecast Performance")
                    st.info(f"Overall MAPE: {overall_accuracy['MAPE'].iloc[0]:.2f}%, Overall Accuracy: {overall_accuracy['Accuracy_Pct'].iloc[0]:.2f}%")

                    # Trend analysis with numeric data
                    if len(accuracy_metrics) >= 3:
                        st.subheader("Error Pattern Analysis")
                        trend = accuracy_numeric.iloc[-3:].mean() - accuracy_numeric.iloc[:3].mean()
                        if trend > WARNING_THRESHOLD:
                            st.success("✓ Forecast accuracy shows an improving trend.")
                        elif trend < -WARNING_THRESHOLD:
                            st.error("⚠ Forecast accuracy shows a declining trend - model may need retraining.")
                        else:
                            st.info("ℹ Forecast accuracy is relatively stable.")

                    # Monthly analysis with numeric data
                    if len(accuracy_metrics) >= 12:
                        month_obj = pd.to_datetime(accuracy_metrics['Month'])
                        month_num = month_obj.dt.month
                        monthly_avgs = pd.DataFrame({'Month_Num': month_num, 'Accuracy': accuracy_numeric}).groupby('Month_Num')['Accuracy'].mean()
                        problem_months = monthly_avgs[monthly_avgs < monthly_avgs.mean() - WARNING_THRESHOLD].index
                        if len(problem_months) > 0:
                            month_names = [calendar.month_name[m] for m in problem_months]
                            st.warning(f"⚠ Lower accuracy detected in: {', '.join(month_names)}. Consider seasonal adjustments.")

                    # Confidence gauge with numeric data
                    st.subheader(f"Forecast Confidence Analysis (for {forecast_period}-month forecast)")
                    confidence = min(95, max(50, accuracy_numeric.mean()))
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={'text': "Forecast Confidence Level"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "red"},
                                {'range': [50, 80], 'color': "orange"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': confidence}
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig)

                    if confidence >= 80:
                        st.success(f"✓ High confidence in the {forecast_period}-month forecast.")
                    elif confidence >= 60:
                        st.info(f"ℹ Moderate confidence in the {forecast_period}-month forecast. Consider shorter horizon.")
                    else:
                        st.error(f"⚠ Low confidence in the {forecast_period}-month forecast. Consider retraining model.")

                except Exception as e:
                    st.error(f"Error analyzing forecast accuracy: {str(e)}")
                    st.error(traceback.format_exc())


def create_enhanced_accuracy_chart(accuracy_df):
    if not pd.api.types.is_datetime64_any_dtype(accuracy_df['Month']):
        accuracy_df['Month'] = pd.to_datetime(accuracy_df['Month'])

    accuracy_df['Accuracy_Pct'] = pd.to_numeric(accuracy_df['Accuracy_Pct'], errors='coerce')
    accuracy_df['Error_Pct'] = pd.to_numeric(accuracy_df['Error_Pct'], errors='coerce')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=accuracy_df['Month'], y=accuracy_df['Accuracy_Pct'],
        name='Accuracy %', mode='lines+markers', line=dict(color='green', width=2)
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=accuracy_df['Month'], y=accuracy_df['Error_Pct'],
        name='Error %', marker_color='rgba(255, 0, 0, 0.6)'
    ), secondary_y=True)

    anomalies = accuracy_df[accuracy_df['Is_Anomaly'] == True]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['Month'], y=anomalies['Accuracy_Pct'],
            mode='markers', marker=dict(symbol='x', size=12, color='red', line=dict(width=2, color='black')),
            name='Anomalies'
        ), secondary_y=False)

        for i, row in anomalies.iterrows():
            fig.add_annotation(
                x=row['Month'], y=row['Accuracy_Pct'],
                text=f"Anomaly: {row['Accuracy_Pct']:.1f}%",
                showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=2, arrowcolor='red', ay=-40
            )

    fig.update_layout(
        title="Forecast Accuracy by Month",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=30, l=30, r=30)
    )

    fig.update_yaxes(title_text="Accuracy %", secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="Error %", secondary_y=True, range=[0, 100])
    fig.update_xaxes(title_text="Month")

    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="orange"),
        y0=80, y1=80, x0=0, x1=1, xref="paper", yref="y"
    )
    fig.add_annotation(
        x=0.01, y=80, xref="paper", yref="y",
        text="Acceptable Threshold (80%)",
        showarrow=False, font=dict(size=10, color="orange")
    )

    return fig


def show_forecasting_deprec():
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
