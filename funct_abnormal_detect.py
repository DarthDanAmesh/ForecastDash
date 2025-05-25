# funct_abnormal_detect.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional
from constants import STANDARD_COLUMNS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def detect_sales_anomalies(df: pd.DataFrame, window: int = 3, sigma: float = 2.0) -> Optional[pd.DataFrame]:
    """Detect SKU-level anomalies in demand data."""
    logger.info(f"Detecting anomalies with window={window}, sigma={sigma})")
    
    if df.empty:
        st.error("Input DataFrame is empty.", icon="🚨")
        logger.error("Empty input DataFrame")
        return None
    
    # Check if the DataFrame is pivoted (material codes as columns, date as index)
    # This typically happens if prepare_time_series was called with a material column
    is_pivoted = not all(col in df.columns for col in [STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['material']]) and df.index.name == STANDARD_COLUMNS['date']

    if is_pivoted:
        logger.info("Input DataFrame appears to be pivoted. Unpivoting...")
        try:
            df_unpivoted = df.reset_index().melt(
                id_vars=[STANDARD_COLUMNS['date']], 
                var_name=STANDARD_COLUMNS['material'], 
                value_name=STANDARD_COLUMNS['demand']
            )
            df = df_unpivoted
            logger.info(f"Unpivoted DataFrame columns: {df.columns.tolist()}")
        except Exception as e:
            st.error(f"Failed to unpivot DataFrame: {str(e)}", icon="🚨")
            logger.error(f"Error unpivoting DataFrame: {str(e)}. Columns were: {df.columns.tolist()}, Index name: {df.index.name}")
            return None

    required_cols = [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['material']]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Invalid input: DataFrame is missing columns: {', '.join(missing_cols)}.", icon="🚨")
        logger.error(f"Missing columns: {missing_cols}. Available columns: {list(df.columns)}")
        return None
    
    try:
        anomalies_list = []
        for material in df[STANDARD_COLUMNS['material']].unique():
            material_df = df[df[STANDARD_COLUMNS['material']] == material].set_index(STANDARD_COLUMNS['date'])
            ts = material_df[STANDARD_COLUMNS['demand']]
            ts = pd.to_numeric(ts, errors='coerce').dropna()
            
            if len(ts) < window:
                logger.warning(f"Skipping material {material}: insufficient data points ({len(ts)} < {window})")
                continue
                
            rolling_mean = ts.rolling(window=window, min_periods=1).mean()
            rolling_std = ts.rolling(window=window, min_periods=1).std().fillna(ts.std())
            upper_bound = rolling_mean + (rolling_std * sigma)
            lower_bound = rolling_mean - (rolling_std * sigma)
            
            material_anomalies = pd.DataFrame({
                STANDARD_COLUMNS['material']: material,
                'date': ts.index,
                'demand': ts,
                'rolling_mean': rolling_mean,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'is_anomaly': (ts > upper_bound) | (ts < lower_bound)
            })
            anomalies_list.append(material_anomalies)
        
        if not anomalies_list:
            st.warning("No valid time series data for anomaly detection.", icon="⚠️")
            logger.info("No anomalies detected: no valid time series data")
            return None
            
        anomalies = pd.concat(anomalies_list, ignore_index=True)
        logger.info(f"Detected {anomalies['is_anomaly'].sum()} anomalies across SKUs")
        return anomalies
    except Exception as e:
        st.error(f"Failed to detect anomalies: {str(e)}.", icon="🚨")
        logger.exception("Anomaly detection error")
        return None

def detect_material_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect anomalies in material demand patterns."""
    material_col = STANDARD_COLUMNS['material']
    date_col = STANDARD_COLUMNS['date']
    demand_col = STANDARD_COLUMNS['demand']
    
    if not all(col in df.columns for col in [material_col, date_col, demand_col]):
        return df
    
    result_df = df.copy()
    result_df['is_anomaly'] = False
    
    # Process each material separately
    for material, group in df.groupby(material_col):
        if len(group) < 5:  # Skip materials with too few data points
            continue
            
        # Calculate rolling mean and standard deviation
        rolling_mean = group[demand_col].rolling(window=3, min_periods=1).mean()
        rolling_std = group[demand_col].rolling(window=3, min_periods=1).std().fillna(group[demand_col].std())
        
        # Mark anomalies (demand > 3 standard deviations from mean)
        threshold = 3
        anomalies = abs(group[demand_col] - rolling_mean) > (threshold * rolling_std)
        
        # Update the result dataframe
        result_df.loc[group.index[anomalies], 'is_anomaly'] = True
    
    return result_df

def plot_anomalies(anomalies: pd.DataFrame, forecast: pd.DataFrame = None, title: str = "SKU Demand Anomaly Detection") -> Optional[go.Figure]:
    """Create SKU-level anomaly plot with optional forecast overlay."""
    if anomalies is None or anomalies.empty:
        st.warning("No anomaly data available to plot.", icon="⚠️")
        return None
    
    try:
        fig = go.Figure()
        materials = anomalies[STANDARD_COLUMNS['material']].unique()
        
        for material in materials:
            material_anomalies = anomalies[anomalies[STANDARD_COLUMNS['material']] == material]
            
            # Demand line
            fig.add_trace(go.Scatter(
                x=material_anomalies['date'],
                y=material_anomalies['demand'],
                name=f'{material} Demand',
                mode='lines',
                line=dict(width=2)
            ))
            
            # Rolling mean
            fig.add_trace(go.Scatter(
                x=material_anomalies['date'],
                y=material_anomalies['rolling_mean'],
                name=f'{material} Rolling Mean',
                line=dict(width=1.5, dash='dash')
            ))
            
            # Bounds
            fig.add_trace(go.Scatter(
                x=material_anomalies['date'].tolist() + material_anomalies['date'].tolist()[::-1],
                y=material_anomalies['upper_bound'].tolist() + material_anomalies['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(44,160,44,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{material} Bounds',
                showlegend=False
            ))
            
            # Anomalies
            anomaly_points = material_anomalies[material_anomalies['is_anomaly']]
            if not anomaly_points.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_points['date'],
                    y=anomaly_points['demand'],
                    mode='markers',
                    name=f'{material} Anomalies',
                    marker=dict(size=10, symbol='x', color='red')
                ))
            
            # Forecast overlay
            if forecast is not None and isinstance(forecast, pd.DataFrame) and not forecast.empty:
                material_forecast = forecast[forecast[STANDARD_COLUMNS['material']] == material]
                if not material_forecast.empty:
                    fig.add_trace(go.Scatter(
                        x=material_forecast['date'],
                        y=material_forecast['forecast'],
                        name=f'{material} Forecast',
                        mode='lines',
                        line=dict(width=2, dash='dot')
                    ))
                    fig.add_trace(go.Scatter(
                        x=material_forecast['date'].tolist() + material_forecast['date'].tolist()[::-1],
                        y=material_forecast['upper_bound'].tolist() + material_forecast['lower_bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{material} Forecast CI',
                        showlegend=False
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Demand",
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig
    except Exception as e:
        st.error(f"Error creating anomaly plot: {str(e)}.", icon="🚨")
        logger.exception("Anomaly plotting error")
        return None

def show_anomaly_detection(data: pd.DataFrame, mode: str = "Simple") -> None:
    """Display SKU-level anomaly detection UI."""
    st.subheader("Demand Anomaly Detection", help="Identify unusual demand patterns per SKU.")
    
    if data is None or data.empty:
        st.warning("No data loaded. Please upload a CSV file.", icon="⚠️")
        return
    
    if mode == "Technical":
        st.markdown("### Tune Anomaly Detection Parameters")
        col1, col2 = st.columns(2)
        with col1:
            window = st.slider(
                "Rolling Window Size",
                min_value=1, max_value=12, value=3,
                help="Number of periods for rolling mean and std."
            )
        with col2:
            sigma = st.slider(
                "Sigma Threshold",
                min_value=1.0, max_value=5.0, value=2.0, step=0.5,
                help="Standard deviations for anomaly bounds."
            )
    else:
        window = 3
        sigma = 2.0
    
    cache_key = f"anomalies_{window}_{sigma}_{data[STANDARD_COLUMNS['material']].nunique()}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = detect_sales_anomalies(data, window, sigma)
    
    anomalies = st.session_state[cache_key]
    forecast = st.session_state.state.forecasts.get("DeepAR")
    
    if anomalies is not None and not anomalies.empty:
        with st.container():
            fig = plot_anomalies(anomalies, forecast)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        if mode == "Simple":
            st.markdown("### Key Anomaly Insights")
            num_anomalies = anomalies['is_anomaly'].sum()
            if num_anomalies > 0:
                st.warning(f"**{num_anomalies} anomalies detected** across SKUs.", icon="⚠️")
                st.markdown("""
                    **Recommended Actions**:
                    - Investigate causes (e.g., promotions, disruptions).
                    - Adjust inventory for affected SKUs.
                """)
            else:
                st.success("No significant anomalies detected.", icon="✅")
        
        if mode == "Technical":
            st.markdown("### Detailed Anomaly Data")
            anomaly_table = anomalies[anomalies['is_anomaly']].copy()
            if not anomaly_table.empty:
                st.dataframe(
                    anomaly_table,
                    use_container_width=True,
                    column_config={
                        "date": st.column_config.DateColumn(),
                        STANDARD_COLUMNS['material']: st.column_config.TextColumn("SKU"),
                        "demand": st.column_config.NumberColumn(format="%.2f"),
                        "rolling_mean": st.column_config.NumberColumn(format="%.2f"),
                        "upper_bound": st.column_config.NumberColumn(format="%.2f"),
                        "lower_bound": st.column_config.NumberColumn(format="%.2f"),
                        "is_anomaly": st.column_config.CheckboxColumn()
                    }
                )
                st.download_button(
                    label="Download Anomaly Data",
                    data=anomaly_table.to_csv(index=False),
                    file_name="sku_anomalies.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.info("No anomalies detected. Adjust parameters if needed.", icon="ℹ️")
    else:
        st.warning("No anomalies detected or data is invalid.", icon="⚠️")