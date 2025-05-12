import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cls_data_preprocessor import DataProcessor
from cls_session_management import SessionState
from cls_forecast_engine import ForecastEngine
from cls_plots_visuals import Visualizer
from consts_model import DEFAULT_FORECAST_PERIOD
from funct_kpi_forecast_metrics import calculate_forecast_accuracy
import logging
import calendar
from typing import Optional, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MIN_DATA_POINTS = 3
ANOMALY_THRESHOLD = 50  # %
WARNING_THRESHOLD = 5  # % change for trends

# Cache forecast generation
@st.cache_data(show_spinner=False)
def cached_generate_forecast(_ts: pd.Series, model_type: str, forecast_period: int, 
                           _model_info: Dict) -> Optional[pd.Series]:
    """Generate and cache forecast for given model and parameters."""
    try:
        if model_type == "XGBoost":
            last_date = _ts.index[-1]
            forecast = ForecastEngine.forecast_xgboost(
                _model_info['model'],
                _model_info['last_values'],
                forecast_period,
                last_date
            )
        elif model_type == "ARIMA":
            forecast = ForecastEngine.forecast_arima(
                _model_info['model'],
                forecast_period
            )
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
        return forecast
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        return None

def show_forecasting():
    # Validate session state
    if not isinstance(st.session_state.state, SessionState):
        st.error("Session state is not properly initialized. Reinitializing...", icon="🚨")
        st.session_state.state = SessionState.get_or_create()
        st.rerun()

    if st.session_state.state.data is None:
        st.error(
            "No data loaded. Please upload a CSV/Excel file with date and demand columns.", 
            icon="🚨"
        )
        return

    if len(st.session_state.state.data) < MIN_DATA_POINTS:
        st.error(
            f"At least {MIN_DATA_POINTS} data points are required for forecasting.", 
            icon="🚨"
        )
        return

    # Initialize session state variables
    if 'current_forecast' not in st.session_state:
        st.session_state.current_forecast = None
    if 'current_ts' not in st.session_state:
        st.session_state.current_ts = None
    if 'forecast_generated' not in st.session_state:
        st.session_state.forecast_generated = False
    if 'forecast_params' not in st.session_state:
        st.session_state.forecast_params = {'model_type': None, 'forecast_period': None}
    
    # Initialize forecasts dictionary
    if not hasattr(st.session_state.state, 'forecasts'):
        st.session_state.state.forecasts = {}

    # Prepare time series
    ts = DataProcessor.prepare_time_series(st.session_state.state.data)
    if ts is None or ts.empty:
        st.error(
            "Failed to prepare time series. Ensure the data has valid date and demand columns.", 
            icon="🚨"
        )
        return
    st.session_state.current_ts = ts

    # UI based on mode
    if st.session_state.mode == "Simple":
        show_simple_forecast(ts)
    else:
        show_technical_forecast(ts)

def show_simple_forecast(ts: pd.Series):
    """Simplified forecasting UI for non-technical users."""
    st.header("Demand Forecast")
    st.markdown("View predicted demand and inventory recommendations.")

    # Check if a model is available
    default_model = "ARIMA" if "ARIMA" in st.session_state.state.models else None
    if not st.session_state.state.models or default_model is None:
        st.info(
            "No trained models available. Click 'Generate Default Forecast' in the Demand Forecast section above to proceed.",
            icon="ℹ️"
        )
        return

    model_type = default_model
    try:
        model_info = st.session_state.state.models[model_type]
    except KeyError:
        st.error(
            f"Model '{model_type}' not found in session state. Please train a model first.", 
            icon="🚨"
        )
        return

    forecast_period = DEFAULT_FORECAST_PERIOD

    if st.session_state.forecast_generated and st.session_state.current_forecast is not None:
        display_forecast(ts, st.session_state.current_forecast)
        return

    """# Button moved to app.py, but keep for compatibility
    if st.button("Generate Forecast", type="primary", help="Generate a demand forecast"):
        with st.spinner("Generating forecast..."):
            forecast = cached_generate_forecast(ts, model_type, forecast_period, model_info)
            if forecast is not None:
                update_forecast_state(model_type, forecast_period, forecast)
                display_forecast(ts, forecast)
                st.success("Forecast generated successfully!", icon="✅")
            else:
                st.error(
                    "Failed to generate forecast. Check data or model settings.", 
                    icon="🚨"
                )"""

def show_technical_forecast(ts: pd.Series):
    """Detailed forecasting UI for technical users."""
    st.header("Demand Forecasting")
    st.markdown("Generate forecasts, tune parameters, and analyze accuracy.")

    tab1, tab2 = st.tabs(["Generate Forecast", "Analyze Accuracy"])

    with tab1:
        if not st.session_state.state.models:
            st.warning(
                "No trained models available. Please train a model in the Model Tuning tab.", 
                icon="⚠️"
            )
            return
        model_type = st.selectbox(
            "Select Model", 
            list(st.session_state.state.models.keys()), 
            key="forecast_model_select"
        )
        model_info = st.session_state.state.models[model_type]
        forecast_period = st.number_input(
            "Forecast Period (months)", 
            min_value=1, 
            max_value=24, 
            value=DEFAULT_FORECAST_PERIOD, 
            key="forecast_period_input"
        )

        params_changed = (
            model_type != st.session_state.forecast_params['model_type'] or 
            forecast_period != st.session_state.forecast_params['forecast_period']
        )
        if params_changed and st.session_state.forecast_generated:
            st.warning(
                "Parameters changed. Click 'Generate Forecast' to update.", 
                icon="⚠️"
            )
            st.session_state.forecast_generated = False

        if forecast_period > 12:
            st.warning(
                "Long forecast periods may reduce accuracy. Consider shorter horizons.", 
                icon="⚠️"
            )
            proceed = st.checkbox("Proceed anyway", key="proceed_checkbox")
            if not proceed:
                return

        if st.session_state.forecast_generated and st.session_state.current_forecast is not None and not params_changed:
            display_forecast(ts, st.session_state.current_forecast)
        elif st.button("Generate Forecast", key="generate_forecast_button"):
            with st.spinner("Generating forecast..."):
                forecast = cached_generate_forecast(ts, model_type, forecast_period, model_info)
                if forecast is not None:
                    update_forecast_state(model_type, forecast_period, forecast)
                    display_forecast(ts, forecast)
                    st.success("Forecast generated successfully!", icon="✅")
                else:
                    st.error(
                        "Failed to generate forecast. Check model or data settings.", 
                        icon="🚨"
                    )

    with tab2:
        if not st.session_state.forecast_generated or st.session_state.current_forecast is None:
            st.warning(
                "No forecast available. Generate a forecast in the 'Generate Forecast' tab.", 
                icon="⚠️"
            )
            return
        analyze_forecast_accuracy(
            ts, 
            st.session_state.forecast_params['model_type'], 
            st.session_state.forecast_params['forecast_period']
        )

def display_forecast(ts: pd.Series, forecast: pd.Series):
    """Display forecast plot and data."""
    st.subheader("Demand Forecast")
    st.plotly_chart(Visualizer.plot_forecast(ts, forecast), use_container_width=True)
    st.write("Forecast Values:")
    st.dataframe(forecast.to_frame(name='Forecast'))
    st.download_button(
        label="Download Forecast",
        data=forecast.to_frame(name='Forecast').to_csv(),
        file_name="forecast.csv",
        mime="text/csv",
        help="Download the forecast data as CSV."
    )

def update_forecast_state(model_type: str, forecast_period: int, forecast: pd.Series):
    """Update session state with forecast data."""
    st.session_state.forecast_params = {'model_type': model_type, 'forecast_period': forecast_period}
    st.session_state.current_forecast = forecast
    st.session_state.forecast_generated = True
    st.session_state.state.forecasts[model_type] = {
        'forecast': forecast,
        'timestamp': pd.Timestamp.now()
    }
    st.session_state.state.last_forecast = {
        'type': model_type,
        'period': forecast_period,
        'data': forecast
    }

@st.cache_data(show_spinner=False)
def cached_analyze_accuracy(_ts: pd.Series, model_type: str, forecast_period: int, 
                          months: int, _model_info: Dict) -> Optional[pd.DataFrame]:
    """Analyze forecast accuracy for a given period."""
    try:
        analysis_data = _ts[-months:] if months < len(_ts) else _ts
        if len(analysis_data) < MIN_DATA_POINTS + 1:
            return None

        accuracy_metrics = pd.DataFrame()
        min_training_size = MIN_DATA_POINTS

        for i in range(min_training_size, len(analysis_data)):
            train_data = analysis_data.iloc[:i]
            test_point = analysis_data.iloc[i]

            if model_type == "XGBoost":
                train_data_subset = analysis_data.iloc[:i]
                df_features = train_data_subset.to_frame()
                target_col = df_features.columns[0]
                features_df = ForecastEngine.create_features(df_features, target_col)
                X_features = features_df.drop(columns=[target_col, 'date'])
                if X_features.empty:
                    logger.warning(f"Empty features at i={i}")
                    continue
                last_features = X_features.iloc[-1].values
                last_date = train_data_subset.index[-1]
                pred = ForecastEngine.forecast_xgboost(
                    _model_info['model'], last_features, 1, last_date
                ).iloc[0]
            elif model_type == "ARIMA":
                pred = ForecastEngine.forecast_arima(_model_info['model'], 1).iloc[0]
            else:
                continue

            actual = test_point
            actuals_series = pd.Series([actual], index=[analysis_data.index[i]])
            forecast_series = pd.Series([pred], index=[analysis_data.index[i]])
            point_accuracy = calculate_forecast_accuracy(actuals_series, forecast_series, level='monthly')
            point_accuracy['Month'] = analysis_data.index[i]
            point_accuracy['Is_Anomaly'] = point_accuracy['Accuracy_Pct'].iloc[0] < ANOMALY_THRESHOLD
            accuracy_metrics = pd.concat([accuracy_metrics, point_accuracy], ignore_index=True)

        accuracy_metrics['Error_Pct'] = 100 - accuracy_metrics['Accuracy_Pct']
        return accuracy_metrics
    except Exception as e:
        logger.error(f"Accuracy analysis failed: {str(e)}")
        return None

def analyze_forecast_accuracy(ts: pd.Series, model_type: str, forecast_period: int):
    """Display forecast accuracy analysis."""
    st.subheader("Historical Forecast Accuracy")
    model_info = st.session_state.state.models[model_type]
    time_options = ["Last 3 months", "Last 6 months", "Last 12 months", "All available data"]
    selected_period = st.selectbox("Select Time Period", time_options, key="time_period_select")
    months = {"Last 3 months": 3, "Last 6 months": 6, "Last 12 months": 12}.get(selected_period, len(ts))
    months = min(months, len(ts))

    if st.button("Analyze Accuracy", key="analyze_accuracy_button"):
        with st.spinner("Analyzing accuracy..."):
            accuracy_metrics = cached_analyze_accuracy(ts, model_type, forecast_period, months, model_info)
            if accuracy_metrics is None or accuracy_metrics.empty:
                st.warning("Insufficient data for accuracy analysis.", icon="⚠️")
                return

            accuracy_numeric = accuracy_metrics['Accuracy_Pct'].copy()
            error_numeric = accuracy_metrics['Error_Pct'].copy()

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Accuracy", f"{accuracy_numeric.mean():.2f}%")
            col2.metric("Average Error", f"{error_numeric.mean():.2f}%")
            col3.metric("Anomalies Detected", f"{accuracy_metrics['Is_Anomaly'].sum()}/{len(accuracy_metrics)}")

            st.subheader("Monthly Forecast Accuracy")
            st.plotly_chart(create_enhanced_accuracy_chart(accuracy_metrics))

            st.subheader("Detailed Metrics")
            display_metrics = accuracy_metrics.copy()
            display_metrics['Month'] = pd.to_datetime(display_metrics['Month']).dt.strftime('%b %Y')
            display_metrics['Accuracy_Pct'] = display_metrics['Accuracy_Pct'].round(2).astype(str) + '%'
            display_metrics['MAPE'] = display_metrics['MAPE'].round(2).astype(str) + '%'
            st.dataframe(
                display_metrics.style.apply(
                    lambda row: ['background-color: #ffcccc' if row['Is_Anomaly'] else '' for _ in row], axis=1
                )
            )

            overall_accuracy = calculate_forecast_accuracy(
                pd.Series(accuracy_metrics['Actual'].values, index=accuracy_metrics['Month']),
                pd.Series(accuracy_metrics['Forecast'].values, index=accuracy_metrics['Month'])
            )
            st.subheader("Overall Performance")
            st.info(
                f"Overall MAPE: {overall_accuracy['MAPE'].iloc[0]:.2f}%, "
                f"Accuracy: {overall_accuracy['Accuracy_Pct'].iloc[0]:.2f}%"
            )

            if len(accuracy_metrics) >= 3:
                st.subheader("Error Pattern Analysis")
                trend = accuracy_numeric.iloc[-3:].mean() - accuracy_numeric.iloc[:3].mean()
                if trend > WARNING_THRESHOLD:
                    st.success("✓ Forecast accuracy improving.")
                elif trend < -WARNING_THRESHOLD:
                    st.error(
                        "⚠ Forecast accuracy declining. Consider retraining.", 
                        icon="⚠️"
                    )
                else:
                    st.info("ℹ Forecast accuracy stable.")

            if len(accuracy_metrics) >= 12:
                st.subheader("Seasonal Analysis")
                month_obj = pd.to_datetime(accuracy_metrics['Month'])
                month_num = month_obj.dt.month
                monthly_avgs = pd.DataFrame({'Month_Num': month_num, 'Accuracy': accuracy_numeric})
                monthly_avgs = monthly_avgs.groupby('Month_Num')['Accuracy'].mean()
                problem_months = monthly_avgs[monthly_avgs < monthly_avgs.mean() - WARNING_THRESHOLD].index
                if problem_months.any():
                    month_names = [calendar.month_name[m] for m in problem_months]
                    st.warning(
                        f"⚠ Lower accuracy in: {', '.join(month_names)}. Consider seasonal adjustments.", 
                        icon="⚠️"
                    )

            st.subheader(f"Forecast Confidence ({forecast_period}-month)")
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

def create_enhanced_accuracy_chart(accuracy_df: pd.DataFrame) -> go.Figure:
    """Create a chart for forecast accuracy analysis."""
    accuracy_df['Month'] = pd.to_datetime(accuracy_df['Month'])
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=accuracy_df['Month'], y=accuracy_df['Accuracy_Pct'],
        name='Accuracy %', mode='lines+markers', line=dict(color='green', width=2)
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=accuracy_df['Month'], y=accuracy_df['Error_Pct'],
        name='Error %', marker_color='rgba(255, 0, 0, 0.6)'
    ), secondary_y=True)

    anomalies = accuracy_df[accuracy_df['Is_Anomaly']]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['Month'], y=anomalies['Accuracy_Pct'],
            mode='markers', marker=dict(symbol='x', size=12, color='red', line=dict(width=2, color='black')),
            name='Anomalies'
        ), secondary_y=False)
        for _, row in anomalies.iterrows():
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
    fig.add_shape(type="line", line=dict(dash="dash", width=1, color="orange"),
                  y0=80, y1=80, x0=0, x1=1, xref="paper", yref="y")
    fig.add_annotation(x=0.01, y=80, xref="paper", yref="y",
                      text="Acceptable Threshold (80%)", showarrow=False, font=dict(size=10, color="orange"))
    return fig