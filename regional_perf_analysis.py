# regional_perf_analysis.py
import plotly.express as px
import pandas as pd
import numpy as np
import logging
from column_config import STANDARD_COLUMNS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_regional_performance(df, value_col=None):
    """
    Analyze performance by region with focus on underperforming areas
    
 dox    Args:
        df (pd.DataFrame): Input dataframe containing sales data
        value_col (str): Column name containing the value to analyze
        
    Returns:
        pd.DataFrame: Processed dataframe with performance metrics by region
    """
    try:
        # Use demand column from column config if value_col not specified
        if value_col is None:
            value_col = STANDARD_COLUMNS['demand']
        
        # Get country column from column config
        country_col = STANDARD_COLUMNS['country']
        
        # Verify required columns exist
        if country_col not in df.columns:
            # Try case-insensitive matching for country column
            country_matches = [col for col in df.columns if col.lower() == country_col.lower()]
            if country_matches:
                country_col = country_matches[0]
            else:
                raise ValueError(f"Country column '{country_col}' not found in dataframe")
        
        if value_col not in df.columns:
            # Try case-insensitive matching for value column
            value_matches = [col for col in df.columns if col.lower() == value_col.lower()]
            if value_matches:
                value_col = value_matches[0]
            else:
                raise KeyError(f"Column '{value_col}' not found in dataframe")
        
        # Ensure value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            try:
                # Log unique values for debugging
                logger.info(f"Unique values in '{value_col}' before conversion: {df[value_col].unique().tolist()}")
                # Try to convert to numeric, coercively (which will turn errors into NaN)
                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                
                # Log and drop problematic rows
                problematic_rows = df[df[value_col].isna()]
                if not problematic_rows.empty:
                    logger.warning(f"Found {len(problematic_rows)} rows with non-numeric values in '{value_col}':\n{problematic_rows}")
                    df = df.dropna(subset=[value_col])
                    logger.info(f"Dropped {len(problematic_rows)} rows with non-numeric values in '{value_col}'")
                
            except Exception as e:
                raise ValueError(f"Could not convert '{value_col}' to numeric: {str(e)}")
        
        # Group by country and calculate key metrics
        region_performance = df.groupby(country_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()
        region_performance.columns = ['Country', 'Total', 'Average', 'Count']
        
        # Calculate percentage of total sales
        total_sales = region_performance['Total'].sum()
        region_performance['Percent_of_Total'] = (region_performance['Total'] / total_sales * 100).round(2)
        
        # Sort by total sales
        region_performance = region_performance.sort_values('Total', ascending=False)
        
        # Calculate expected share based on order count distribution
        region_performance['Expected_Share'] = (region_performance['Count'] / region_performance['Count'].sum() * 100).round(2)
        
        # Calculate performance gap
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