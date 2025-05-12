# regional_perf_analysis.py
import plotly.express as px
import pandas as pd
import numpy as np
import logging
from column_config import STANDARD_COLUMNS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def analyze_regional_performance(df, value_col='demand'):
    """
    Analyze performance by region using standardized column keys.
    """
    try:
        country_col = 'country'

        if country_col not in df.columns:
            raise ValueError(f"Country column '{country_col}' not found in dataframe")
        if value_col not in df.columns:
            raise KeyError(f"Column '{value_col}' not found in dataframe")

        # Ensure value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            logger.info(f"Unique values in '{value_col}' before conversion: {df[value_col].unique().tolist()}")
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            problematic_rows = df[df[value_col].isna()]
            if not problematic_rows.empty:
                logger.warning(f"Dropping {len(problematic_rows)} non-numeric rows from '{value_col}'")
                df = df.dropna(subset=[value_col])

        # Group and analyze
        region_performance = df.groupby(country_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()
        region_performance.columns = ['Country', 'Total', 'Average', 'Count']
        total_sales = region_performance['Total'].sum()
        region_performance['Percent_of_Total'] = (region_performance['Total'] / total_sales * 100).round(2)
        region_performance['Expected_Share'] = (region_performance['Count'] / region_performance['Count'].sum() * 100).round(2)
        region_performance['Performance_Gap'] = (region_performance['Percent_of_Total'] - region_performance['Expected_Share']).round(2)

        return region_performance

    except Exception as e:
        raise RuntimeError(f"Error in regional performance analysis: {str(e)}")



def plot_regional_performance(region_performance):
    """
    Create visualizations for regional performance analysis
    
    Args:
        region_performance (pd.DataFrame): DataFrame with performance metrics by region
        
    Returns:
        tuple: Two Plotly figure objects
    """
    try:
        # Bar chart of total sales by region with color gradient based on performance gap
        fig1 = px.bar(
            region_performance, 
            x='Country', 
            y='Total', 
            title="Total Sales by Region", 
            color='Performance_Gap',
            color_continuous_scale=['red', 'yellow', 'green'],
            labels={'Total': 'Total Sales', 'Country': 'Region'}
        )
        fig1.update_layout(xaxis_tickangle=-45, height=400)
        
        # Performance gap visualization
        fig2 = px.bar(
            region_performance, 
            x='Country', 
            y='Performance_Gap',
            title="Performance Gap by Region (Actual % - Expected %)",
            color='Performance_Gap', 
            color_continuous_scale=['red', 'yellow', 'green'],
            labels={'Performance_Gap': 'Performance Gap (%)', 'Country': 'Region'}
        )
        fig2.update_layout(xaxis_tickangle=-45, height=400)
        
        return fig1, fig2
    
    except Exception as e:
        raise RuntimeError(f"Error creating regional performance plots: {str(e)}")