# simple_mode_ui.py
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from typing import Optional, Dict, Any
from cls_data_preprocessor import DataProcessor
from forecast_engine import ForecastEngine
from funct_shw_forecast_plot import display_forecast
from renders_for_simple_mode_ui import render_mobile_ui, render_mobile_controls, detect_mobile
from constants import STANDARD_COLUMNS, DEFAULT_PREDICTION_LENGTH, COUNTRY_CODE_MAP, MAX_ENCODER_LENGTH, DEFAULT_FORECAST_PERIOD, DEFAULT_FREQ
from simple_mode_utils import render_adjustment_wizard
import streamlit.components.v1 as components
import logging
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_simple_mode():
    """Render Simple Mode UI for non-technical users."""
    if st.session_state.state.data is None:
        st.error("No data loaded. Please upload a CSV file in the sidebar.", icon="🚨")
        return

    is_mobile = detect_mobile()
    
    if is_mobile:
        render_mobile_ui()
        render_mobile_controls()
        # Optionally, add chatbot for mobile too, might need specific layout
        # render_chatbot_interface() 
    else:
        col_main, col_right = st.columns([4, 1])
        with col_main:
            render_central_controls()
            render_data_table()
            render_forecast_section()
            #render_adjustment_wizard()
        with col_right:
            #render_sku_panel()
            render_chatbot_interface()


def render_filters(): # This function is now primarily for mobile or can be inlined if only used once
    """Render filters for material and country."""
    # st.markdown("<div class='sidebar-filter'>", unsafe_allow_html=True) # Context might change
    # st.subheader("Filters") # Subheader might be redundant if called within render_central_controls
    
    data = st.session_state.state.data
    material_col = STANDARD_COLUMNS['material']
    country_col = STANDARD_COLUMNS['country']
    
    material_options = ['All'] + sorted(data[material_col].unique().tolist()) if material_col in data.columns and data is not None else ['All']
    country_options = ['All'] + sorted(data[country_col].unique().tolist()) if country_col in data.columns and data is not None else ['All']
    
    selected_materials = st.multiselect(
        "Select SKUs/Materials",
        material_options,
        default=['All'],
        help="Filter by product SKU or material code.",
        key="material_filter"
    )
    selected_countries = st.multiselect(
        "Select Countries",
        country_options,
        default=['All'],
        help="Filter by country or location.",
        key="country_filter"
    )
    
    if 'All' in selected_materials:
        selected_materials = material_options[1:] if len(material_options) > 1 else []
    if 'All' in selected_countries:
        selected_countries = country_options[1:] if len(country_options) > 1 else []
    
    st.session_state.filters = {
        'materials': selected_materials,
        'countries': selected_countries
    }
    # st.markdown("</div>", unsafe_allow_html=True)


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply filters to data based on session state."""
    if data is None:
        return pd.DataFrame()
        
    filters_state = st.session_state.get('filters', {'materials': [], 'countries': []})
    filtered_data = data.copy()
    
    # Apply material filter
    materials_to_filter = filters_state.get('materials', [])
    if materials_to_filter and STANDARD_COLUMNS['material'] in filtered_data.columns:
        filtered_data = filtered_data[filtered_data[STANDARD_COLUMNS['material']].isin(materials_to_filter)]
    
    # Apply country filter
    countries_to_filter = filters_state.get('countries', [])
    if countries_to_filter and STANDARD_COLUMNS['country'] in filtered_data.columns:
        filtered_data = filtered_data[filtered_data[STANDARD_COLUMNS['country']].isin(countries_to_filter)]
    
    # Apply date range filter, ensuring enough historical data
    if 'date_range' in st.session_state and st.session_state.date_range:
        date_range = st.session_state.date_range
        if len(date_range) == 2 and STANDARD_COLUMNS['date'] in filtered_data.columns:
            start_date, end_date = date_range
            try:
                data_dates = pd.to_datetime(filtered_data[STANDARD_COLUMNS['date']])
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)

                if data_dates.dt.tz is not None:
                    start_date_dt = start_date_dt.tz_localize(data_dates.dt.tz) if start_date_dt.tz is None else start_date_dt.tz_convert(data_dates.dt.tz)
                    end_date_dt = end_date_dt.tz_localize(data_dates.dt.tz) if end_date_dt.tz is None else end_date_dt.tz_convert(data_dates.dt.tz)
                
                # Extend start_date backward to include context for forecasting
                context_start_date = start_date_dt - pd.offsets.MonthBegin(MAX_ENCODER_LENGTH)
                filtered_data = filtered_data[
                    (data_dates >= context_start_date) &
                    (data_dates <= end_date_dt)
                ]
            except Exception as e:
                logger.error(f"Error applying date filter: {e}")
                st.warning(f"Could not apply date filter: {e}")
    
    return filtered_data

def render_central_controls():
    """Render date range selector, forecast freeze, and data filters."""
    st.subheader("Demand Planning Controls")
    data = st.session_state.state.data
    date_col = STANDARD_COLUMNS['date']
    
    if data is None:
        st.error("Dataset not loaded. Please upload a valid file.", icon="🚨")
        return

    # --- Date Range and Freeze Forecast --- 
    col1, col2 = st.columns(2)
    with col1:
        if date_col not in data.columns:
            st.error(f"Dataset missing date column: '{date_col}'. Please upload a valid file.", icon="🚨")
        else:
            try:
                min_date = pd.to_datetime(data[date_col]).min()
                max_date = pd.to_datetime(data[date_col]).max()
                st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Choose the time period for analysis.",
                    key="date_range"
                )
            except Exception as e:
                st.error(f"Invalid date column or data: {str(e)}.", icon="🚨")
    
    with col2:
        st.checkbox(
            "Freeze Forecast",
            value=st.session_state.get('freeze_forecast', False), # Ensure key exists
            help="Lock the forecast to prevent updates until unfrozen.",
            key="freeze_forecast"
        )
    
    st.markdown("---_Filters_---") # Separator for filters

    # --- Material and Country Filters (moved from render_filters) ---
    material_col = STANDARD_COLUMNS['material']
    country_col = STANDARD_COLUMNS['country']
    
    # Use columns for better layout of filters
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        material_options = ['All'] + sorted(data[material_col].unique().tolist()) if material_col in data.columns and not data[material_col].empty else ['All']
        selected_materials = st.multiselect(
            "Select SKUs/Materials",
            material_options,
            default=['All'] if 'All' in material_options else material_options[1:] if len(material_options) > 1 else [],
            help="Filter by product SKU or material code.",
            key="material_filter_central"
        )

    with filter_col2:
        country_options = ['All'] + sorted(data[country_col].unique().tolist()) if country_col in data.columns and not data[country_col].empty else ['All']
        selected_countries = st.multiselect(
            "Select Countries",
            country_options,
            default=['All'] if 'All' in country_options else country_options[1:] if len(country_options) > 1 else [],
            help="Filter by country or location.",
            key="country_filter_central"
        )
    
    # Update session state filters (ensure keys are consistent or update logic in filter_data)
    processed_materials = []
    if 'All' in selected_materials:
        processed_materials = material_options[1:] if len(material_options) > 1 else []
    else:
        processed_materials = selected_materials

    processed_countries = []
    if 'All' in selected_countries:
        processed_countries = country_options[1:] if len(country_options) > 1 else []
    else:
        processed_countries = selected_countries

    st.session_state.filters = {
        'materials': processed_materials,
        'countries': processed_countries
    }

def render_data_table():
    """Render data overview in tabs with selection: Monthly Demand, Demand by Country, Top SKUs, and SKU Demand Trend."""
    st.subheader("Data Overview")
    
    data = filter_data(st.session_state.state.data)
    if data is None or data.empty:
        st.warning("No data matches the selected filters or no data loaded.", icon="⚠️")
        return

    required_cols = [STANDARD_COLUMNS['date'], STANDARD_COLUMNS['demand'], STANDARD_COLUMNS['material']]
    if not all(col in data.columns for col in required_cols):
        st.error(f"Data is missing one or more required columns: {', '.join(required_cols)}.", icon="🚨")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Monthly Demand", "Demand by Country", "Top SKUs", "SKU Demand Trend"])

    with tab1:
        with st.expander("📊 Monthly Demand Overview", expanded=False):
            try:
                monthly_data = data.groupby([pd.Grouper(key=STANDARD_COLUMNS['date'], freq='ME'), 
                                        STANDARD_COLUMNS['material']])[STANDARD_COLUMNS['demand']].sum().reset_index()
                monthly_data[STANDARD_COLUMNS['date']] = pd.to_datetime(monthly_data[STANDARD_COLUMNS['date']]).dt.strftime('%Y-%m')
                
                st.dataframe(
                    monthly_data, 
                    use_container_width=True, 
                    key="monthly_demand_df", 
                    on_select="rerun", 
                    selection_mode=["multi-row", "multi-column"],
                    height=400
                )
                
                # Store selection
                if st.session_state.get("monthly_demand_df"):
                    st.session_state.data_table_selection = st.session_state.monthly_demand_df.selection

            except Exception as e:
                logger.error(f"Error displaying monthly demand table: {str(e)}")
                st.error(f"Failed to display monthly demand: {str(e)}.", icon="🚨")

    with tab2:
        st.markdown("#### Demand by Country")
        if STANDARD_COLUMNS['country'] in data.columns:
            try:
                country_summary = data.groupby(STANDARD_COLUMNS['country'])[STANDARD_COLUMNS['demand']].sum().reset_index()
                country_summary = country_summary[country_summary[STANDARD_COLUMNS['country']].isin(COUNTRY_CODE_MAP.keys())]
                country_summary['country_full_name'] = country_summary[STANDARD_COLUMNS['country']].map(COUNTRY_CODE_MAP)
                
                if country_summary.empty:
                    st.info("No data available for specified countries after filtering.")
                else:
                    # For plots, selection is not directly applicable in the same way as dataframes
                    # If you need to select from the data used for the plot, display it as a dataframe too
                    fig = px.bar(
                        country_summary,
                        x='country_full_name',
                        y=STANDARD_COLUMNS['demand'],
                        title="Demand by Country (Filtered)",
                        labels={STANDARD_COLUMNS['demand']: 'Total Demand', 'country_full_name': 'Country'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    # Optionally, display the country_summary dataframe for selection
                    st.dataframe(country_summary, key="country_summary_df", on_select="rerun", selection_mode=["multi-row"])
                    if st.session_state.get("country_summary_df"):
                        st.session_state.data_table_selection = st.session_state.country_summary_df.selection
                    if st.session_state.get("country_summary_df"):
                        st.session_state.data_table_selection = st.session_state.country_summary_df.selection

            except Exception as e:
                logger.error(f"Error creating country bar plot: {str(e)}")
                st.error(f"Failed to create country bar plot: {str(e)}.", icon="🚨")
        else:
            st.info("Country data is not available in the dataset.")

    with tab3:
        st.markdown("#### Top 5 SKUs by Demand")
        try:
            sku_summary = data.groupby(STANDARD_COLUMNS['material'])[STANDARD_COLUMNS['demand']].sum().nlargest(5).reset_index()
            # Similar to tab2, if selection is needed from this data, display it as a dataframe
            fig = px.bar(
                sku_summary,
                x=STANDARD_COLUMNS['material'],
                y=STANDARD_COLUMNS['demand'],
                title="Top 5 SKUs by Demand",
                labels={STANDARD_COLUMNS['demand']: 'Total Demand'}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sku_summary, key="top_skus_df", on_select="rerun", selection_mode=["multi-row"])
            if st.session_state.get("top_skus_df"):
                st.session_state.data_table_selection = st.session_state.top_skus_df.selection

        except Exception as e:
            logger.error(f"Error creating SKU bar plot: {str(e)}")
            st.error(f"Failed to create SKU bar plot: {str(e)}.", icon="🚨")

    with tab4:
        st.markdown("#### SKU Demand Trend")
        material_col = STANDARD_COLUMNS['material']
        demand_col = STANDARD_COLUMNS['demand']
        date_col = STANDARD_COLUMNS['date']

        if not all(col in data.columns for col in [material_col, demand_col, date_col]):
            st.warning("Data is missing required columns (material, demand, or date) for SKU trend view.", icon="⚠️")
        else:
            try:
                # Ensure date column is datetime
                data_trend = data.copy()
                data_trend[date_col] = pd.to_datetime(data_trend[date_col])

                sku_demand_history = data_trend.groupby([
                    material_col,
                    pd.Grouper(key=date_col, freq='ME') 
                ])[demand_col].sum().reset_index()

                sku_history_list = sku_demand_history.groupby(material_col)[demand_col].apply(list).reset_index(name='demand_history')
                sku_total_demand = data_trend.groupby(material_col)[demand_col].sum().reset_index(name='total_demand')

                if sku_history_list.empty:
                    sku_summary_df_trend = sku_total_demand
                    sku_summary_df_trend['demand_history'] = [[] for _ in range(len(sku_total_demand))]
                else:
                    sku_summary_df_trend = pd.merge(sku_total_demand, sku_history_list, on=material_col, how='left')
                    sku_summary_df_trend['demand_history'] = sku_summary_df_trend['demand_history'].apply(lambda x: x if isinstance(x, list) else [])

                # --- Integration of AI Adjusted Forecast ---
                sku_summary_df_trend['adjusted_demand_trend'] = [[] for _ in range(len(sku_summary_df_trend))] # Initialize with empty lists
                max_adjusted_demand_history = 0

                if hasattr(st.session_state.state, 'forecasts') and "DeepAR" in st.session_state.state.forecasts and not st.session_state.state.forecasts["DeepAR"].empty:
                    adjusted_forecast_df = st.session_state.state.forecasts["DeepAR"].copy()
                    if not adjusted_forecast_df.empty and material_col in adjusted_forecast_df.columns and 'forecast' in adjusted_forecast_df.columns:
                        adjusted_forecast_df[date_col] = pd.to_datetime(adjusted_forecast_df[date_col])
                        
                        # Group adjusted forecast by material to get a list of forecast values
                        adjusted_trend_data = adjusted_forecast_df.groupby(material_col)['forecast'].apply(list).reset_index(name='adjusted_demand_trend_temp')
                        
                        # Merge with sku_summary_df_trend
                        sku_summary_df_trend = pd.merge(sku_summary_df_trend, adjusted_trend_data, on=material_col, how='left')
                        # Fill NaNs from merge (for SKUs with historical but no adjusted forecast) with empty lists
                        sku_summary_df_trend['adjusted_demand_trend'] = sku_summary_df_trend['adjusted_demand_trend_temp'].apply(lambda x: x if isinstance(x, list) else [])
                        sku_summary_df_trend.drop(columns=['adjusted_demand_trend_temp'], inplace=True, errors='ignore')

                        # Calculate max for y-axis scaling
                        for history in sku_summary_df_trend['adjusted_demand_trend']:
                            if history:
                                max_adjusted_demand_history = max(max_adjusted_demand_history, max(history) if history else 0)
                # --- End Integration ---

                sku_summary_df_trend.rename(columns={material_col: 'SKU', 'total_demand': 'Total Demand'}, inplace=True)

                if sku_summary_df_trend.empty:
                    st.info("No SKU data to display after processing for trends.")
                else:
                    max_demand_history = 0
                    if 'demand_history' in sku_summary_df_trend.columns:
                        for history in sku_summary_df_trend['demand_history']:
                            if history:
                                max_demand_history = max(max_demand_history, max(history) if history else 0)
                    
                    # Determine y_max for charts, considering both historical and adjusted trends
                    y_max_chart = max(10, max_demand_history, max_adjusted_demand_history)

                    column_config = {
                        "SKU": st.column_config.TextColumn("SKU Code"),
                        "Total Demand": st.column_config.NumberColumn(
                            "Total Demand",
                            help="Total demand for the SKU over the selected period",
                            format="%.2f"
                        ),
                        "demand_history": st.column_config.LineChartColumn(
                            "Demand Trend (Monthly)", 
                            help="Monthly historical demand trend for the SKU",
                            y_min=0, 
                            y_max=int(y_max_chart * 1.1) 
                        ),
                    }

                    # Add adjusted demand trend column if data exists
                    if 'adjusted_demand_trend' in sku_summary_df_trend.columns and sku_summary_df_trend['adjusted_demand_trend'].apply(lambda x: bool(x)).any():
                        column_config["adjusted_demand_trend"] = st.column_config.TextColumn(
                            "AI Adjusted Trend Data",
                            help="Monthly AI-adjusted demand data points for the SKU"
                        )

                    st.dataframe(
                        sku_summary_df_trend,
                        column_config=column_config,
                        use_container_width=True,
                        hide_index=True,
                        key="sku_trend_df", # Unique key for this dataframe
                        on_select="rerun",
                        selection_mode=["multi-row", "multi-column"] 
                    )
                    if st.session_state.get("sku_trend_df"):
                        st.session_state.data_table_selection = st.session_state.sku_trend_df.selection
            
            except Exception as e:
                logger.error(f"Error creating SKU demand trend view: {str(e)}")
                st.error(f"Failed to create SKU demand trend view: {str(e)}.", icon="🚨")

    # Display current selection (for debugging or context)
    # st.write("Current DataFrame Selection:", st.session_state.get('data_table_selection'))


def render_forecast_section():
    """Render forecast generation and visualization section."""
    st.subheader("Demand Forecast")
    if st.session_state.get('freeze_forecast', False):
        st.warning("Forecast is frozen. Uncheck 'Freeze Forecast' to generate a new forecast.", icon="⚠️")
        return
    
    # Check if filters have changed since last forecast
    if st.session_state.get('forecast_generated', False) and filters_changed():
        st.warning("Filters have changed since the last forecast. Please regenerate the forecast for updated results.", icon="⚠️")
        if st.button("Regenerate Forecast", type="primary"):
            # Reset forecast_generated to trigger new forecast
            st.session_state.forecast_generated = False
            st.rerun()
    
    if st.session_state.get('forecast_generated', False) and "DeepAR" in st.session_state.state.forecasts:
        display_forecast(st.session_state.state.data, st.session_state.state.forecasts["DeepAR"])
        return
    
    if st.button("Generate Forecast", type="primary", help="Generate a demand forecast"):
        with st.spinner("Generating forecast..."):
            data = filter_data(st.session_state.state.data)
            if data.empty:
                st.error("No data after filtering. Adjust filters.", icon="🚨")
                return
            try:
                # Get the end date from the filtered data or date range
                date_col = STANDARD_COLUMNS['date']
                data[date_col] = pd.to_datetime(data[date_col])
                if 'date_range' in st.session_state and st.session_state.date_range and len(st.session_state.date_range) == 2:
                    end_date = pd.to_datetime(st.session_state.date_range[1])
                else:
                    end_date = data[date_col].max()

                # Generate forecast for 6 months
                forecast_result = ForecastEngine.forecast(data, forecast_horizon=DEFAULT_FORECAST_PERIOD, freq=DEFAULT_FREQ)
                if forecast_result is None:
                    st.error("Failed to generate forecast.", icon="🚨")
                    return
                forecast_df = forecast_result['forecast']
                if forecast_df is None or forecast_df.empty:
                    st.error("Generated forecast is empty.", icon="🚨")
                    return
                
                # Ensure forecast dates are monthly and start after end_date
                forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                forecast_df = forecast_df[forecast_df['date'] > end_date]
                
                st.session_state.state.forecasts["DeepAR"] = forecast_df
                st.session_state.forecast_generated = True
                # Save current filter state for future comparison
                save_filter_state()
                
                logger.info(f"Forecast DataFrame columns: {list(forecast_df.columns)}")
                display_forecast(data, forecast_df)
                st.success("Forecast generated successfully!", icon="✅")
            except Exception as e:
                logger.error(f"Forecast error: {str(e)}")
                st.error(f"Failed to generate forecast: {str(e)}.", icon="🚨")

def render_sku_panel():
    """Render right-side panel with selected SKUs, quantities, and demand history, with selection."""
    st.markdown("<div class='sidebar-filter'>", unsafe_allow_html=True)
    st.subheader("Selected SKUs Overview (Select for context)")
    
    data = filter_data(st.session_state.state.data) 
    if data is None or data.empty:
        st.info("No SKUs selected or data available for selected filters.", icon="ℹ️")
        return
    
    material_col = STANDARD_COLUMNS['material']
    demand_col = STANDARD_COLUMNS['demand']
    date_col = STANDARD_COLUMNS['date']

    if not all(col in data.columns for col in [material_col, demand_col, date_col]):
        st.warning("Data is missing required columns (material, demand, or date) for SKU panel.", icon="⚠️")
        return

    try:
        data[date_col] = pd.to_datetime(data[date_col])
    except Exception as e:
        logger.error(f"Error converting date column in render_sku_panel: {e}")
        st.warning(f"Could not process date column: {e}", icon="⚠️")
        return

    sku_demand_history = data.groupby([
        material_col,
        pd.Grouper(key=date_col, freq='ME') 
    ])[demand_col].sum().reset_index()

    sku_history_list = sku_demand_history.groupby(material_col)[demand_col].apply(list).reset_index(name='demand_history')
    sku_total_demand = data.groupby(material_col)[demand_col].sum().reset_index(name='total_demand')

    if sku_history_list.empty:
        sku_summary_df = sku_total_demand
        sku_summary_df['demand_history'] = [[] for _ in range(len(sku_total_demand))]
    else:
        sku_summary_df = pd.merge(sku_total_demand, sku_history_list, on=material_col, how='left')
        sku_summary_df['demand_history'] = sku_summary_df['demand_history'].apply(lambda x: x if isinstance(x, list) else [])

    sku_summary_df.rename(columns={material_col: 'SKU', 'total_demand': 'Total Demand'}, inplace=True)

    if sku_summary_df.empty:
        st.info("No SKU data to display after processing.")
        return

    max_demand_history = 0
    if 'demand_history' in sku_summary_df.columns:
        for history in sku_summary_df['demand_history']:
            if history:
                max_demand_history = max(max_demand_history, max(history))
    y_max_chart = max(10, max_demand_history)

    # Make the dataframe selectable
    event = st.dataframe(
        sku_summary_df,
        column_config={
            "SKU": st.column_config.TextColumn("SKU Code"),
            "Total Demand": st.column_config.NumberColumn(
                "Total Demand",
                help="Total demand for the SKU over the selected period",
                format="%.2f"
            ),
            "demand_history": st.column_config.LineChartColumn(
                "Demand Trend (Monthly)", 
                help="Monthly demand trend for the SKU",
                y_min=0, 
                y_max=int(y_max_chart * 1.1)
            ),
        },
        use_container_width=True,
        hide_index=True,
        key="sku_panel_df", # Unique key for this dataframe
        on_select="rerun",
        selection_mode=["multi-row"] # Or other modes as needed
    )
    
    # Store selection
    if st.session_state.sku_panel_df:
        st.session_state.sku_panel_selection = st.session_state.sku_panel_df.selection
    
    # Display current selection (for debugging or context)
    # st.write("Current SKU Panel Selection:", st.session_state.sku_panel_selection)

    # The Adjust Stock button was commented out in your context, restore if needed
    # st.button("Adjust Stock", help="Initiate stock adjustment for selected SKUs.", key="adjust_stock_panel")
    st.markdown("</div>", unsafe_allow_html=True)

def render_feedback_widget():
    """Render floating feedback widget."""
    st.markdown("### 💬 Feedback")
    
    # Feedback button with popover
    with st.popover("📝 Send Feedback", help="Help us improve forecasts"):
        st.markdown("**Share your insights:**")
        
        # Feedback form
        feedback_type = st.selectbox(
            "Feedback Type",
            ["Forecast Accuracy", "Feature Request", "Bug Report", "General"]
        )
        
        selected_sku = st.selectbox(
            "Related SKU (optional)",
            ["None"] + list(st.session_state.state.data[STANDARD_COLUMNS['material']].unique()) 
            if st.session_state.state.data is not None and STANDARD_COLUMNS['material'] in st.session_state.state.data.columns 
            else ["None"]
        )
        
        comment = st.text_area(
            "Comments",
            placeholder="Describe your feedback...",
            max_chars=500
        )
        
        if st.button("Submit Feedback", type="primary"):
            collect_feedback(
                user_id=st.session_state.get('user_id', 'anonymous'),
                forecast_id=st.session_state.get('current_forecast_id', 'unknown'),
                feedback_type=feedback_type,
                sku=selected_sku if selected_sku != "None" else None,
                comment=comment
            )

def collect_feedback(user_id: str, forecast_id: str, feedback_type: str, sku: Optional[str], comment: str):
    """Collect and store user feedback."""
    if 'feedback' not in st.session_state:
        st.session_state.feedback = []
    
    feedback_entry = {
        "user_id": user_id,
        "forecast_id": forecast_id,
        "feedback_type": feedback_type,
        "sku": sku,
        "comment": comment,
        "timestamp": pd.Timestamp.now(),
        "status": "new"
    }
    
    st.session_state.feedback.append(feedback_entry)
    st.success("✅ Feedback submitted! Thank you for helping us improve.", icon="🙏")
    
    # Notify data scientists (in a real implementation, this would send an alert)
    if len(st.session_state.feedback) % 5 == 0:  # Every 5th feedback
        st.info(f"📊 {len(st.session_state.feedback)} feedback entries collected. Data scientists have been notified.", icon="📈")


def chat_stream(prompt: str, selection_context: Dict[str, Any]):
    """Generates a response from the Ollama API using a streaming connection."""
    context_str = "No specific data selection provided by the user for this query."
    if selection_context and (selection_context.get('rows') or selection_context.get('columns')):
        # Be mindful of context length limits for the LLM
        context_str = f"The user has selected the following data context: Rows {selection_context.get('rows')}, Columns {selection_context.get('columns')}. Please consider this context if relevant to the query."

    # Check if the user is asking about the chatbot's identity
    identity_keywords = ["who are you", "what are you", "tell me about yourself", "your name"]
    if any(keyword in prompt.lower() for keyword in identity_keywords):
        yield "I am Depli. A useful Franke Demand Planning Chatbot powered by IBM WatsonX ready to provide Q/A for forecasts and data"
        return # Stop processing after providing the introduction

    full_prompt = f"{context_str}\n\nUser Query: {prompt}"

    ollama_api_url = "http://localhost:11434/api/chat" # Default Ollama API endpoint for chat
    # You can change the model name to any qwen2 model you have pulled, e.g., "qwen2:0.5b", "qwen2:1.5b", "qwen2:7b" or "qwen2:latest"
    model_name = "qwen2:0.5b" # Using a smaller Qwen2 model as an example

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant analyzing demand and forecast data."},
            {"role": "user", "content": full_prompt}
        ],
        "stream": True
    }

    try:
        with requests.post(ollama_api_url, json=payload, stream=True) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors
            for line in response.iter_lines():
                if line:
                    try:
                        json_chunk = json.loads(line.decode('utf-8'))
                        if 'message' in json_chunk and 'content' in json_chunk['message']:
                            content_piece = json_chunk['message']['content']
                            yield content_piece
                        if json_chunk.get("done"):
                            break # Stream finished
                    except json.JSONDecodeError:
                        logger.warning(f"Ollama stream: Could not decode JSON line: {line}")
                        continue # Skip malformed lines
                    except Exception as e:
                        logger.error(f"Error processing Ollama stream chunk: {e}")
                        yield f"Error: {e}"
                        break
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        yield f"Error: Could not connect to Ollama or the model. Please ensure Ollama is running and the model '{model_name}' is available. Details: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred in chat_stream: {e}")
        yield f"Error: An unexpected error occurred. {e}"


def save_feedback(feedback_text: str, rating: str, chat_history: list):
    # Check if the feedback key exists before accessing
    feedback_key = f"feedback_{index}"
    if feedback_key in st.session_state:
        st.session_state.history[index]["feedback"] = st.session_state[feedback_key]
    else:
        logger.warning(f"Feedback key {feedback_key} not found in session_state during save_feedback.")

def render_chatbot_interface():
    # Add custom CSS for inspired design
    st.markdown(
        """
        <style>
        /* Gradient header for the chatbot */
        .chatbot-header {
            background: linear-gradient(90deg, #6B48FF, #00DDEB);
            color: white;
            padding: 10px 20px;
            border-radius: 10px 10px 0 0;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 18px;
            font-weight: bold;
        }
        .chatbot-header img {
            width: 24px;
            height: 24px;
        }
        /* Chat container */
        .chat-container {
            border: 1px solid #e0e0e0;
            border-radius: 0 0 10px 10px;
            padding: 20px;
            background: #f9f9f9;
            max-height: 500px;
            overflow-y: auto;
        }
        /* User and assistant messages */
        .stChatMessage.user {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 10px;
            margin: 10px 0;
            max-width: 70%;
            margin-left: auto;
            border: 1px solid #e0e0e0;
        }
        .stChatMessage.assistant {
            background-color: #E6E6FA;
            border-radius: 15px;
            padding: 10px;
            margin: 10px 0;
            max-width: 70%;
            margin-right: auto;
            border: 1px solid #d0d0ff;
        }
        /* Chat input styling */
        div[data-testid="stChatInput"] {
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            background: #ffffff;
            padding: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Chatbot header
    st.markdown(
        """
        <div class="chatbot-header">
            <img src="https://img.icons8.com/ios-filled/50/ffffff/bot.png" alt="Bot Icon"/>
            Depli Chatbot
        </div>
        """,
        unsafe_allow_html=True
    )

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for i, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                feedback = message.get("feedback", None)
                feedback_session_key = f"feedback_{i}"
                if feedback_session_key not in st.session_state:
                    st.session_state[feedback_session_key] = feedback
                
                st.feedback(
                    "thumbs",
                    key=feedback_session_key,
                    on_change=save_feedback,
                    args=[i],
                )

    # Consolidate selection context
    active_selection_context = None
    if st.session_state.get('sku_panel_selection') and (st.session_state.sku_panel_selection.get('rows') or st.session_state.sku_panel_selection.get('columns')):
        active_selection_context = st.session_state.sku_panel_selection
    elif st.session_state.get('data_table_selection') and (st.session_state.data_table_selection.get('rows') or st.session_state.data_table_selection.get('columns')):
        active_selection_context = st.session_state.data_table_selection

    # Chat input
    if prompt := st.chat_input("Ask about the data or forecast..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response_content = st.write_stream(chat_stream(prompt, active_selection_context))
            
            # Simulate a graph in the response (placeholder for now)
            st.markdown("📊 *Click here to see Graph.*")
            
            # Feedback for the new message
            new_message_index = len(st.session_state.history)
            feedback_key_new = f"feedback_{new_message_index}"
            st.session_state[feedback_key_new] = None
            st.feedback(
                "thumbs",
                key=feedback_key_new,
                on_change=save_feedback,
                args=[new_message_index],
            )
        st.session_state.history.append({"role": "assistant", "content": response_content, "feedback": None})

    st.markdown('</div>', unsafe_allow_html=True)


# Add this function to track filter state when forecast is generated
def save_filter_state():
    """Save the current filter state for comparison."""
    st.session_state.last_forecast_filters = {
        'materials': st.session_state.filters.get('materials', []).copy() if isinstance(st.session_state.filters.get('materials', []), list) else [],
        'countries': st.session_state.filters.get('countries', []).copy() if isinstance(st.session_state.filters.get('countries', []), list) else [],
        'date_range': st.session_state.date_range if 'date_range' in st.session_state else None
    }

# Add this function to check if filters have changed
def filters_changed():
    """Check if filters have changed since last forecast generation."""
    if 'last_forecast_filters' not in st.session_state:
        return False
    
    last_filters = st.session_state.last_forecast_filters
    current_materials = st.session_state.filters.get('materials', [])
    current_countries = st.session_state.filters.get('countries', [])
    current_date_range = st.session_state.date_range if 'date_range' in st.session_state else None
    
    # Compare materials
    if set(last_filters['materials']) != set(current_materials):
        return True
    
    # Compare countries
    if set(last_filters['countries']) != set(current_countries):
        return True
    
    # Compare date ranges
    if last_filters['date_range'] != current_date_range:
        return True
    
    return False