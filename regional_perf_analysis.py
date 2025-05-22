# regional_perf_analysis.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from constants import STANDARD_COLUMNS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_regional_performance(df: pd.DataFrame, forecast: pd.DataFrame = None) -> pd.DataFrame:
    """Analyze performance by region and SKU."""
    try:
        required_cols = [STANDARD_COLUMNS['country'], STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['material']]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {', '.join(set(required_cols) - set(df.columns))}")
        
        # Country-level performance
        region_performance = df.groupby(STANDARD_COLUMNS['country'])[STANDARD_COLUMNS['demand']].agg(['sum', 'mean', 'count']).reset_index()
        region_performance.columns = ['country', 'Total_Demand', 'Average_Demand', 'Record_Count']
        
        # SKU-level performance per country
        sku_performance = df.groupby([STANDARD_COLUMNS['country'], STANDARD_COLUMNS['material']])[STANDARD_COLUMNS['demand']].sum().reset_index()
        sku_performance = sku_performance.rename(columns={STANDARD_COLUMNS['demand']: 'Total_Demand'})
        
        # Forecast accuracy (if forecast provided)
        if forecast is not None and not forecast.empty:
            forecast_agg = forecast.groupby([STANDARD_COLUMNS['country'], STANDARD_COLUMNS['material']])['forecast'].sum().reset_index()
            actual_agg = df.groupby([STANDARD_COLUMNS['country'], STANDARD_COLUMNS['material']])[STANDARD_COLUMNS['demand']].sum().reset_index()
            merged = forecast_agg.merge(actual_agg, on=[STANDARD_COLUMNS['country'], STANDARD_COLUMNS['material']], suffixes=('_forecast', '_actual'))
            merged['Accuracy'] = 100 * (1 - abs(merged['forecast'] - merged[STANDARD_COLUMNS['demand']]) / merged[STANDARD_COLUMNS['demand']])
            sku_performance = sku_performance.merge(
                merged[[STANDARD_COLUMNS['country'], STANDARD_COLUMNS['material'], 'Accuracy']],
                on=[STANDARD_COLUMNS['country'], STANDARD_COLUMNS['material']],
                how='left'
            )
        
        return {'region': region_performance, 'sku': sku_performance}
    except Exception as e:
        logger.error(f"Regional performance analysis failed: {str(e)}")
        raise RuntimeError(f"Error in regional performance analysis: {str(e)}")

def plot_regional_performance(performance: dict) -> tuple[go.Figure, go.Figure]:
    """Create visualizations for regional and SKU performance."""
    try:
        region_performance = performance['region']
        sku_performance = performance['sku']
        
        # Region bar chart
        fig1 = px.bar(
            region_performance,
            x='country',
            y='Total_Demand',
            title="Total Demand by Region",
            labels={'Total_Demand': 'Total Demand'},
            color='Average_Demand',
            color_continuous_scale='Viridis'
        )
        fig1.update_layout(xaxis_tickangle=-45, height=400)
        
        # SKU heatmap
        fig2 = px.treemap(
            sku_performance,
            path=[STANDARD_COLUMNS['country'], STANDARD_COLUMNS['material']],
            values='Total_Demand',
            title="SKU Demand by Region",
            color='Accuracy' if 'Accuracy' in sku_performance.columns else 'Total_Demand',
            color_continuous_scale='RdYlGn' if 'Accuracy' in sku_performance.columns else 'Blues'
        )
        fig2.update_layout(height=600)
        
        return fig1, fig2
    except Exception as e:
        logger.error(f"Error creating regional performance plots: {str(e)}")
        raise RuntimeError(f"Error creating regional performance plots: {str(e)}")