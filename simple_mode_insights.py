import streamlit as st
import pandas as pd
from column_config import STANDARD_COLUMNS
from funct_detect_prod_discontinued import detect_discontinued_products

import altair as alt

def display_top_skus():
    """Highlight top 3 SKUs by growth rate with enhanced UX, tooltips, and optional SKU list."""
    data = st.session_state.state.data
    material_col = STANDARD_COLUMNS['material']
    date_col = STANDARD_COLUMNS['date']
    demand_col = STANDARD_COLUMNS['demand']

    if material_col not in data.columns or date_col not in data.columns:
        st.warning("Material or date column missing for SKU analysis.", icon="⚠️")
        return

    data = data.dropna(subset=[material_col, date_col, demand_col])
    data['year'] = pd.to_datetime(data[date_col]).dt.year

    # Compute yearly demand per SKU
    yearly_demand = (
        data.groupby([material_col, 'year'])[demand_col]
        .sum()
        .unstack(fill_value=0)
    )

    if yearly_demand.shape[1] < 2:
        st.info("Not enough historical data (minimum 2 years required) to compute growth rates.")
        return

    # Ensure we are using the latest two consecutive years
    years = sorted(yearly_demand.columns)
    if len(years) < 2:
        st.info("At least two full years of data are needed to calculate growth rate.")
        return

    # Make sure the latest year is truly the last in the dataset
    last_year = years[-1]
    prev_year = years[-2]

    # Growth rate calculation
    yearly_demand['Growth_Rate'] = (
        (yearly_demand[last_year] - yearly_demand[prev_year]) /
        yearly_demand[prev_year].replace(0, pd.NA) * 100
    ).fillna(0)

    # Filter out rows where both years have zero demand
    yearly_demand = yearly_demand[
        ~((yearly_demand[prev_year] == 0) & (yearly_demand[last_year] == 0))
    ]

    # Get top 3 SKUs by growth
    top_skus = yearly_demand.nlargest(3, 'Growth_Rate').reset_index()

    # Handle case where all SKUs are declining
    # Get top 3 SKUs by growth rate (even if all are negative)
    top_skus = yearly_demand.sort_values(by='Growth_Rate', ascending=False).head(3).reset_index()

    # Check if all SKUs are declining or flat
    all_negative_or_flat = (yearly_demand['Growth_Rate'] <= 0).all()

    # Display subheader depending on trend
    if all_negative_or_flat:
        st.subheader("📉 Least Declining SKUs")
        st.info("All SKUs showed flat or declining demand growth. Showing least declining SKUs.")
    else:
        st.subheader("Top Performing SKUs")

    # Tooltip explanation for growth calculation
    with st.popover("ℹ️ How is growth rate calculated?"):
        st.markdown("""
        Growth Rate is calculated as:

        $$
        \\text{Growth Rate (\\%)} = \\left( \\frac{\\text{Current Year Demand} - \\text{Previous Year Demand}}{\\text{Previous Year Demand}} \\right) \\times 100
        $$

        This gives the percentage change in demand from one year to the next.
        """)

    # Display each top SKU
    for _, row in top_skus.iterrows():
        sku = row[material_col]
        growth = row['Growth_Rate']
        ly_val = row.get(last_year, 0)
        py_val = row.get(prev_year, 0)

        delta_color = "normal" if growth >= 0 else "inverse"

        col1, col2 = st.columns([1.5, 2])

        with col1:
            st.markdown(f"**{sku}**")
            st.metric(label="Growth Rate", value=f"{growth:.1f}%", delta=f"{growth:.1f}%", delta_color=delta_color)
            st.caption(f"{prev_year}: {py_val:.0f} → {last_year}: {ly_val:.0f}")

        with col2:
            chart_df = pd.DataFrame({
                'Year': [str(prev_year), str(last_year)],
                'Demand': [py_val, ly_val]
            })

            bar_chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('Year:O', axis=alt.Axis(title=None)),
                y=alt.Y('Demand:Q'),
                tooltip=['Year', 'Demand'],
                color=alt.condition(
                    alt.datum.Demand > 0,
                    alt.value("#1f77b4") if growth >= 0 else alt.value("#d62728"),
                    alt.value("#999")
                )
            ).properties(height=80)

            st.altair_chart(bar_chart, use_container_width=True)

    # Toggle to show full list of SKUs
    show_all = st.checkbox("📊 Show all SKUs with growth rates")

    if show_all:
        st.dataframe(
            yearly_demand[[prev_year, last_year, 'Growth_Rate']]
            .sort_values(by='Growth_Rate', ascending=False)
            .style.format({"Growth_Rate": "{:.1f}%"})
        )


def display_discontinued_products():
    """Flag products to discontinue based on low demand or obsolescence."""
    st.subheader("Products to Discontinue")

    data = st.session_state.state.data
    material_col = STANDARD_COLUMNS['material']
    date_col = STANDARD_COLUMNS['date']


    if material_col not in data.columns or date_col not in data.columns:
        st.warning("Required columns missing for discontinuation analysis.", icon="⚠️")
        return

    discontinued = detect_discontinued_products(data, threshold_months=3)

    show_all = st.checkbox("Show all flagged SKUs")

    if show_all:
        st.dataframe(discontinued[[material_col, 'Months_Since_Last_Order']])

    if discontinued.empty:
        st.info("✅ No products flagged for discontinuation.", icon="ℹ️")
        return

    for _, row in discontinued.head(5).iterrows():
        st.warning(
            f"**{row[material_col]}**: No orders for {int(row['Months_Since_Last_Order'])} months.",
            icon="⚠️"
        )

def display_recommendation_cards():
    """Show actionable recommendation cards with quantity and timing suggestions."""
    st.subheader("Recommendations")
    data = st.session_state.state.data
    material_col = STANDARD_COLUMNS['material']
    date_col = STANDARD_COLUMNS['date']
    demand_col = STANDARD_COLUMNS['demand']

    if any(col not in data.columns for col in [material_col, date_col, demand_col]):
        return

    # Preprocess
    data = data.dropna(subset=[material_col, date_col, demand_col])
    data[date_col] = pd.to_datetime(data[date_col])
    latest_month = data[date_col].max().strftime("%B %Y")

    # Get monthly average demand per SKU
    monthly_avg = data.groupby([material_col, data[date_col].dt.to_period("M")])[demand_col].sum().groupby(level=0).mean()
    
    # Get top/bottom performers
    top_skus = monthly_avg.nlargest(2)
    bottom_skus = monthly_avg.nsmallest(2)

    for sku, demand in top_skus.items():
        increase_pct = 30  # Suggest +30% increase for high demand
        st.success(
            f"**Increase stock** for `{sku}` by **+{increase_pct}%** for **{latest_month}** due to sustained high demand.",
            icon="✅"
        )

    for sku, demand in bottom_skus.items():
        reduce_pct = 20  # Suggest supply check/reduction
        st.warning(
            f"**Review supply** for `{sku}` and consider reducing stock by **~{reduce_pct}%** for **{latest_month}** due to weak demand.",
            icon="⚠️"
        )
