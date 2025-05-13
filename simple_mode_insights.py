import streamlit as st
import pandas as pd
from column_config import STANDARD_COLUMNS
from funct_detect_prod_discontinued import detect_discontinued_products

def display_top_skus():
    """Highlight top 3 SKUs by growth rate."""
    st.subheader("Top Performing SKUs")
    data = st.session_state.state.data
    material_col = STANDARD_COLUMNS['material']
    if material_col not in data.columns:
        st.warning("Material column missing for SKU analysis.", icon="⚠️")
        return
    
    # Calculate growth rate (example: year-over-year demand change)
    data['year'] = pd.to_datetime(data[STANDARD_COLUMNS['date']]).dt.year
    yearly_demand = data.groupby([material_col, 'year'])[STANDARD_COLUMNS['demand']].sum().unstack()
    yearly_demand['Growth_Rate'] = (
        (yearly_demand[yearly_demand.columns[-1]] - yearly_demand[yearly_demand.columns[-2]]) /
        yearly_demand[yearly_demand.columns[-2]] * 100
    ).fillna(0)
    
    top_skus = yearly_demand.nlargest(3, 'Growth_Rate').reset_index()
    for _, row in top_skus.iterrows():
        st.info(f"**{row[material_col]}**: Growth Rate {row['Growth_Rate']:.2f}%", icon="📈")

def display_discontinued_products():
    """Flag products to discontinue based on low demand or obsolescence."""
    st.subheader("Products to Discontinue")
    data = st.session_state.state.data
    material_col = STANDARD_COLUMNS['material']
    if material_col not in data.columns:
        st.warning("Material column missing for discontinuation analysis.", icon="⚠️")
        return
    
    discontinued = detect_discontinued_products(data, threshold=3)
    if discontinued.empty:
        st.info("No products flagged for discontinuation.", icon="ℹ️")
        return
    
    for _, row in discontinued.head(3).iterrows():
        st.warning(
            f"**{row['material']}**: No orders for {row['Months_Since_Last_Order']} months.",
            icon="⚠️"
        )

def display_recommendation_cards():
    """Show actionable recommendation cards."""
    st.subheader("Recommendations")
    data = st.session_state.state.data
    material_col = STANDARD_COLUMNS['material']
    if material_col not in data.columns:
        return
    
    # Example recommendations based on demand trends
    high_demand_skus = data.groupby(material_col)[STANDARD_COLUMNS['demand']].mean().nlargest(2).index
    for sku in high_demand_skus:
        st.success(f"**Increase stock** for {sku} due to high demand.", icon="✅")
    
    low_demand_skus = data.groupby(material_col)[STANDARD_COLUMNS['demand']].mean().nsmallest(2).index
    for sku in low_demand_skus:
        st.warning(f"**Review supply** for {sku} due to low demand.", icon="⚠️")