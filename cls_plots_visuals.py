import streamlit as st
import pandas as pd
import plotly.express as px

class Visualizer:
    @staticmethod
    def plot_time_series(ts, title="Time Series"):
        fig = px.line(ts, title=title)
        fig.update_layout(xaxis_title='Date', yaxis_title='Value')
        return fig

    @staticmethod
    def plot_forecast(history, forecast, title="Demand Forecast"):
        fig = px.line(history, title=title)
        fig.add_scatter(x=forecast.index, y=forecast.values, name='Forecast')
        fig.update_layout(xaxis_title='Date', yaxis_title='Demand')
        return fig

    @staticmethod
    def plot_geographical(df, location_col='Country Key Ship-to', value_col='Delivery Quantity'):
        try:
            # Ensure value column is numeric
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            
            # Group by country and sum the values
            country_counts = df.groupby(location_col)[value_col].sum().reset_index()
            
            # Convert country codes to uppercase if they aren't already
            country_counts[location_col] = country_counts[location_col].str.upper()
            
            # Create a mapping of common country codes to full names if needed
            country_code_map = {
                'IT': 'Italy',
                'DE': 'Germany',
                'FR': 'France',
                'ES': 'Spain',
                'GB': 'United Kingdom',
                'NL': 'Netherlands',
                'BE': 'Belgium',
                'PT': 'Portugal',
                'CH': 'Switzerland',
                'AT': 'Austria',
                'SE': 'Sweden',
                'NO': 'Norway',
                'FI': 'Finland',
                'DK': 'Denmark',
                'IE': 'Ireland',
                'PL': 'Poland',
                'CZ': 'Czech Republic',
                'HU': 'Hungary',
                'RO': 'Romania',
                'GR': 'Greece',
                'TR': 'Turkey',
            }
            
            # Apply the mapping if the codes are in our dictionary
            country_counts['country_name'] = country_counts[location_col].map(country_code_map)
            
            # Use the mapped names if available, otherwise use the codes
            location_values = country_counts['country_name'].fillna(country_counts[location_col])
            
            # Create the choropleth map
            fig = px.choropleth(country_counts, 
                                locations=location_values,
                                locationmode='country names',
                                color=value_col,
                                scope='europe', 
                                hover_name=location_values,
                                hover_data={location_col: True, value_col: True},
                                color_continuous_scale='Viridis',
                                title=f"Delivery Quantity by Country")
            
            # Adjust layout for better visibility
            fig.update_layout(
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    center=dict(lat=54, lon=15),
                    projection_type='equirectangular'
                ),
                margin={"r":0,"t":30,"l":0,"b":0}
            )
            
            return fig
        
        except Exception as e:
            st.error(f"Error creating geographical plot: {str(e)}")
            # Return empty figure if there's an error
            return px.choropleth()
            

    @staticmethod
    def plot_product_performance(df, group_col='Material', value_col='Delivery Quantity', top_n=10):
        try:
            # Convert to numeric and drop non-numeric values
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df = df.dropna(subset=[value_col, group_col])
            
            # Group and sum
            grouped = df.groupby(group_col)[value_col].sum()
            
            # Convert to DataFrame and sort
            top_products = grouped.reset_index().sort_values(value_col, ascending=False).head(top_n)
            
            fig = px.bar(top_products, x=group_col, y=value_col, 
                        title=f"Top {top_n} {group_col}s by {value_col}")
            return fig
        except Exception as e:
            st.error(f"Error creating product performance plot: {str(e)}")
            return px.bar()