import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def detect_sales_anomalies(ts: pd.Series, window: int = 3, sigma: float = 2.0) -> Optional[pd.DataFrame]:
    """
    Detect anomalies in a demand time series using rolling statistics.

    Args:
        ts (pd.Series): Time series data (e.g., demand values indexed by date).
        window (int): Rolling window size for calculating mean and std (default: 3).
        sigma (float): Number of standard deviations to define anomaly bounds (default: 2.0).

    Returns:
        Optional[pd.DataFrame]: DataFrame with columns ['demand', 'rolling_mean', 'upper_bound', 'lower_bound', 'is_anomaly'],
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
        # Ensure numeric data
        ts = pd.to_numeric(ts, errors='coerce')
        if ts.isna().any():
            st.warning("Some values in the time series are non-numeric and were converted to NaN. Dropping invalid rows.")
            ts = ts.dropna()

        # Calculate rolling statistics
        rolling_mean = ts.rolling(window=window, min_periods=1).mean()
        rolling_std = ts.rolling(window=window, min_periods=1).std().fillna(ts.std())

        # Define upper and lower bounds
        upper_bound = rolling_mean + (rolling_std * sigma)
        lower_bound = rolling_mean - (rolling_std * sigma)

        # Create anomalies DataFrame with standardized column names
        anomalies = pd.DataFrame({
            'demand': ts,
            'rolling_mean': rolling_mean,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'is_anomaly': (ts > upper_bound) | (ts < lower_bound)
        })

        logger.info(f"Detected {anomalies['is_anomaly'].sum()} anomalies")
        return anomalies

    except Exception as e:
        st.error(f"Failed to detect anomalies: {str(e)}. Ensure your data has valid numeric values and a datetime index.")
        logger.exception("Anomaly detection error")
        return None

def plot_anomalies(anomalies: pd.DataFrame, title: str = "Demand Anomaly Detection") -> Optional[go.Figure]:
    """
    Create an interactive Plotly plot of the time series with anomalies highlighted.

    Args:
        anomalies (pd.DataFrame): DataFrame with columns ['demand', 'rolling_mean', 'upper_bound', 'lower_bound', 'is_anomaly'].
        title (str): Plot title (default: "Demand Anomaly Detection").

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
            y='demand',
            title=title,
            labels={'demand': 'Demand', 'index': 'Date'},
            color_discrete_sequence=['#1f77b4']  # Consistent color for demand line
        )

        # Add rolling mean
        fig.add_scatter(
            x=anomalies.index,
            y=anomalies['rolling_mean'],
            name='Rolling Mean',
            line=dict(color='#ff7f0e', width=1.5),
        )

        # Add upper and lower bounds
        fig.add_scatter(
            x=anomalies.index,
            y=anomalies['upper_bound'],
            name='Upper Bound',
            line=dict(color='#2ca02c', dash='dash', width=1),
        )
        fig.add_scatter(
            x=anomalies.index,
            y=anomalies['lower_bound'],
            name='Lower Bound',
            line=dict(color='#2ca02c', dash='dash', width=1),
            fill='tonexty',
            fillcolor='rgba(44, 160, 44, 0.1)',
        )

        # Highlight anomalies
        anomaly_points = anomalies[anomalies['is_anomaly']]
        fig.add_scatter(
            x=anomaly_points.index,
            y=anomaly_points['demand'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='#d62728', size=10, symbol='x'),
        )

        # Customize layout for accessibility and clarity
        fig.update_layout(
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=14, family='Arial'),
            xaxis_title="Date",
            yaxis_title="Demand",
            showlegend=True,
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa',
        )

        # Ensure high contrast for accessibility
        fig.update_traces(
            hoverlabel=dict(bgcolor='white', font_size=12, font_color='black')
        )

        logger.info("Anomaly plot generated successfully")
        return fig

    except Exception as e:
        st.error(f"Error creating anomaly plot: {str(e)}. Please verify your data.")
        logger.exception("Anomaly plotting error")
        return None

def show_anomaly_detection(data: pd.DataFrame, mode: str = "Simple") -> None:
    """
    Display the anomaly detection section in the Streamlit app, tailored for both user types.

    Args:
        data (pd.DataFrame): Input DataFrame with demand data.
        mode (str): User mode ('Simple' or 'Technical') to customize the UI.
    """
    st.subheader(
        "Demand Anomaly Detection",
        help="Identify unusual demand patterns that may require investigation, such as sudden spikes or drops."
    )

    if data is None or data.empty:
        st.warning("No data loaded. Please upload a CSV or Excel file in the sidebar.")
        return

    from cls_data_preprocessor import DataProcessor

    # Prepare time series
    ts = DataProcessor.prepare_time_series(data)
    if ts is None:
        st.error(
            "Unable to prepare time series. Ensure your data has valid 'date' and 'demand' columns. "
            f"Available columns: {', '.join(data.columns.tolist())}"
        )
        return

    # Technical Mode: Parameter tuning with real-time feedback
    if mode == "Technical":
        st.markdown(
            "### Tune Anomaly Detection Parameters",
            help="Adjust the rolling window and sigma threshold to fine-tune anomaly detection sensitivity."
        )
        col1, col2 = st.columns(2)
        with col1:
            window = st.slider(
                "Rolling Window Size",
                min_value=1,
                max_value=12,
                value=3,
                key="anomaly_window",
                help="Number of periods to calculate rolling mean and standard deviation. Larger windows smooth out short-term fluctuations."
            )
        with col2:
            sigma = st.slider(
                "Sigma Threshold",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="anomaly_sigma",
                help="Number of standard deviations to define anomaly bounds. Higher values detect only extreme anomalies."
            )
    else:
        window = 3
        sigma = 2.0

    # Cache key for dynamic inputs in Technical Mode
    cache_key = f"anomalies_{window}_{sigma}_{ts.to_json()}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = detect_sales_anomalies(ts, window=window, sigma=sigma)

    anomalies = st.session_state[cache_key]

    if anomalies is not None:
        # Display plot in a card-like container
        with st.container():
            st.markdown(
                '<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; background-color: #ffffff;">',
                unsafe_allow_html=True
            )
            fig = plot_anomalies(anomalies)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Simple Mode: Key insights for non-technical users
        if mode == "Simple":
            st.markdown(
                '<div class="tooltip">ℹ️<span class="tooltiptext">Anomalies indicate unusual demand patterns that may affect inventory planning.</span></div> '
                '### Key Anomaly Insights',
                unsafe_allow_html=True
            )
            num_anomalies = anomalies['is_anomaly'].sum()
            if num_anomalies > 0:
                st.warning(
                    f"**{num_anomalies} anomalies detected** in demand data. These may indicate unexpected spikes or drops.",
                    icon="⚠️"
                )
                st.markdown(
                    """
                    **Recommended Actions**:
                    - Investigate potential causes (e.g., promotions, supply chain disruptions).
                    - Adjust inventory levels to mitigate risks.
                    """,
                    help="Click the ℹ️ icon for more information on anomalies."
                )
            else:
                st.success(
                    "No significant anomalies detected in the demand data. Demand patterns appear stable.",
                    icon="✅"
                )

        # Technical Mode: Detailed table and download option
        if mode == "Technical":
            st.markdown(
                '<div class="tooltip">ℹ️<span class="tooltiptext">View and download detailed anomaly data for further analysis.</span></div> '
                '### Detailed Anomaly Data',
                unsafe_allow_html=True
            )
            anomaly_table = anomalies[anomalies['is_anomaly']].copy()
            if not anomaly_table.empty:
                # Format dates for display
                anomaly_table.index = anomaly_table.index.strftime('%Y-%m-%d')
                anomaly_table = anomaly_table.reset_index().rename(
                    columns={
                        'index': 'Date',
                        'demand': 'Demand',
                        'rolling_mean': 'Rolling Mean',
                        'upper_bound': 'Upper Bound',
                        'lower_bound': 'Lower Bound',
                        'is_anomaly': 'Is Anomaly'
                    }
                )
                st.dataframe(
                    anomaly_table,
                    use_container_width=True,
                    column_config={
                        "Date": st.column_config.DateColumn(),
                        "Demand": st.column_config.NumberColumn(format="%.2f"),
                        "Rolling Mean": st.column_config.NumberColumn(format="%.2f"),
                        "Upper Bound": st.column_config.NumberColumn(format="%.2f"),
                        "Lower Bound": st.column_config.NumberColumn(format="%.2f"),
                        "Is Anomaly": st.column_config.CheckboxColumn()
                    }
                )
                st.download_button(
                    label="Download Anomaly Data",
                    data=anomaly_table.to_csv(index=False),
                    file_name="demand_anomalies.csv",
                    mime="text/csv",
                    type="primary",
                    help="Download a CSV file containing details of detected anomalies."
                )
            else:
                st.info(
                    "No anomalies detected with the current parameters. Try adjusting the window size or sigma threshold.",
                    icon="ℹ️"
                )