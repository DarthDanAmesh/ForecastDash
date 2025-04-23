
import plotly.express as px

def analyze_regional_performance(df, value_col='Delivery Quantity'):
    """Analyze performance by region with focus on underperforming areas"""
    region_performance = df.groupby('Country Key Ship-to')[value_col].agg(['sum', 'mean', 'count']).reset_index()
    region_performance.columns = ['Country', 'Total', 'Average', 'Count']
    
    # Calculate percentage of total
    total_sales = region_performance['Total'].sum()
    region_performance['Percent_of_Total'] = (region_performance['Total'] / total_sales * 100).round(2)
    
    # Sort by total sales
    region_performance = region_performance.sort_values('Total', ascending=False)
    
    # Calculate quota performance (placeholder - replace with actual quota data)
    # For demonstration, we'll assume quota is proportional to count
    region_performance['Expected_Share'] = (region_performance['Count'] / region_performance['Count'].sum() * 100).round(2)
    region_performance['Performance_Gap'] = (region_performance['Percent_of_Total'] - region_performance['Expected_Share']).round(2)
    
    return region_performance

def plot_regional_performance(region_performance):
    """Create visualization for regional performance analysis"""
    # Bar chart of total sales by region
    fig1 = px.bar(region_performance, x='Country', y='Total', 
                 title="Total Sales by Region", color='Performance_Gap',
                 color_continuous_scale=['red', 'yellow', 'green'])
    
    # Performance gap visualization
    fig2 = px.bar(region_performance, x='Country', y='Performance_Gap',
                 title="Performance Gap by Region (Actual % - Expected %)",
                 color='Performance_Gap', color_continuous_scale=['red', 'yellow', 'green'])
    
    return fig1, fig2