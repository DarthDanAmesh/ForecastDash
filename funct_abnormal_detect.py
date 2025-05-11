import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def detect_sales_anomalies(ts: pd.Series, window: int = 3, sigma: float = 2.0) -> Optional[pd.DataFrame]:
    """
    Detect anomalies in a demand time series using rolling statistics.

    Args:
        ts (pd.Series): Time series data (e.g., demand values indexed by date).
        window (int): Rolling window size for calculating mean and std (default: 3).
        sigma (float): Number of standard deviations to define anomaly bounds (default: 2.0).

    Returns:
        Optional[pd.DataFrame]: DataFrame with original data, rolling stats, and anomaly flags,
                               or None if detection fails.
    """
    logger.info(f"Detecting anomalies with window={window}, sigma={sigma}")

    if not isinstance(ts, pd.Series):
        st.error("Invalid input: Time series must be a pandas Series with a datetime index.")
        logger.error("Input is not a pandas Series")
        return None

    if ts.empty or ts.isna().all():
        st.error("Time series is empty or contains only missing values. Please check your data.")
        logger.error("Empty or invalid time series")
        return None

    try:
        # Calculate rolling statistics
        rolling_mean = ts.rolling(window=window, min_periods=1).mean()
        rolling_std = ts.rolling(window=window, min_periods=1).std().fillna(ts.std())

        # Define upper and lower bounds
        upper_bound = rolling_mean + (rolling_std * sigma)
        lower_bound = rolling_mean - (rolling_std * sigma)

        # Create anomalies DataFrame
        anomalies = pd.DataFrame({
            'Demand': ts,
            'Rolling_Mean': rolling_mean,
            'Upper_Bound': upper_bound,
            'Lower_Bound': lower_bound,
            'Is_Anomaly': (ts > upper_bound) | (ts < lower_bound)
        })

        logger.info(f"Detected {anomalies['Is_Anomaly'].sum()} anomalies")
        return anomalies

    except Exception as e:
        st.error(f"Error detecting anomalies: {str(e)}. Please verify your data and try again.")
        logger.exception("Anomaly detection error")
        return None

def plot_anomalies(anomalies: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create an interactive plot of the time series with anomalies highlighted.

    Args:
        anomalies (pd.DataFrame): DataFrame with time series data and anomaly flags.

    Returns:
        Optional[go.Figure]: Plotly figure or None if plotting fails.
    """
    if anomalies is None or anomalies.empty:
        st.warning("No anomaly data available to plot.")
        logger.warning("No data to plot for anomalies")
        return None

    try:
        # Create base line plot for demand
        fig = px.line(
            anomalies,
            x=anomalies.index,
            y='Demand',
            title="Demand Anomaly Detection",
            labels={'Demand': 'Demand', 'index': 'Date'},
        )

        # Add rolling mean
        fig.add_scatter(
            x=anomalies.index,
            y=anomalies['Rolling_Mean'],
            name='Rolling Mean',
            line=dict(color='orange', width=1.5),
        )

        # Add upper and lower bounds
        fig.add_scatter(
            x=anomalies.index,
            y=anomalies['Upper_Bound'],
            name='Upper Bound',
            line=dict(color='green', dash='dash', width=1),
        )
        fig.add_scatter(
            x=anomalies.index,
            y=anomalies['Lower_Bound'],
            name='Lower Bound',
            line=dict(color='green', dash='dash', width=1),
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.1)',
        )

        # Highlight anomalies
        anomaly_points = anomalies[anomalies['Is_Anomaly']]
        fig.add_scatter(
            x=anomaly_points.index,
            y=anomaly_points['Demand'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='x'),
        )

        # Customize layout
        fig.update_layout(
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=14),
            xaxis_title="Date",
            yaxis_title="Demand",
            showlegend=True,
        )

        logger.info("Anomaly plot generated successfully")
        return fig

    except Exception as e:
        st.error(f"Error creating anomaly plot: {str(e)}")
        logger.exception("Anomaly plotting error")
        return None

def show_anomaly_detection():
    """
    Display the anomaly detection section in the Streamlit app, tailored for both user types.
    """
    st.subheader("Demand Anomaly Detection", help="Identify unusual demand patterns that may require investigation.")

    if st.session_state.state.data is None:
        st.warning("No data loaded. Please upload data in the sidebar.")
        return

    from cls_data_preprocessor import DataProcessor
    df = st.session_state.state.data

    # Prepare time series
    ts = DataProcessor.prepare_time_series(df)
    if ts is None:
        st.error(
            "Unable to prepare time series. Ensure your data has valid 'date' and 'demand' columns. "
            "Available columns: " + ", ".join(df.columns.tolist())
        )
        return

    # Technical Mode: Parameter tuning
    if st.session_state.mode == "Technical":
        st.markdown("### Tune Anomaly Detection Parameters")
        col1, col2 = st.columns(2)
        with col1:
            window = st.slider(
                "Rolling Window Size",
                min_value=1,
                max_value=12,
                value=3,
                help="Number of periods to calculate rolling mean and standard deviation."
            )
        with col2:
            sigma = st.slider(
                "Sigma Threshold",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Number of standard deviations to define anomaly bounds."
            )
    else:
        window = 3
        sigma = 2.0

    # Detect anomalies
    anomalies = detect_sales_anomalies(ts, window=window, sigma=sigma)

    if anomalies is not None:
        # Display plot
        fig = plot_anomalies(anomalies)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Simple Mode: Key insights for non-technical users
        if st.session_state.mode == "Simple":
            st.markdown("### Key Anomaly Insights")
            num_anomalies = anomalies['Is_Anomaly'].sum()
            if num_anomalies > 0:
                st.warning(
                    f"Detected {num_anomalies} anomalies in demand data. "
                    "These may indicate unexpected spikes or drops in demand."
                )
                st.markdown(
                    "- **Action**: Investigate anomalies for potential causes (e.g., promotions, supply chain issues).",
                    help="Anomalies highlight unusual demand patterns that may affect inventory planning."
                )
            else:
                st.success("No significant anomalies detected in the demand data.")

        # Technical Mode: Detailed table and download option
        if st.session_state.mode == "Technical":
            st.markdown("### Detailed Anomaly Data")
            anomaly_table = anomalies[anomalies['Is_Anomaly']].reset_index()
            if not anomaly_table.empty:
                anomaly_table.columns = ['Date', 'Demand', 'Rolling Mean', 'Upper Bound', 'Lower Bound', 'Is Anomaly']
                st.dataframe(anomaly_table, use_container_width=True)
                st.download_button(
                    label="Download Anomaly Data",
                    data=anomaly_table.to_csv(index=False),
                    file_name="demand_anomalies.csv",
                    mime="text/csv",
                    help="Download details of detected anomalies as a CSV file."
                )
            else:
                st.info("No anomalies detected with the current parameters.")