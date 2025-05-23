# simple_mode_utils.py
import pandas as pd
from constants import STANDARD_COLUMNS
import streamlit as st
import numpy as np
from typing import Dict, Optional

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter data based on sidebar selections."""
    if data is None or 'filters' not in st.session_state:
        return data
    
    filtered_data = data.copy()
    filters = st.session_state.filters
    
    if filters['materials']:
        filtered_data = filtered_data[filtered_data[STANDARD_COLUMNS['material']].isin(filters['materials'])]
    
    if filters['countries']:
        filtered_data = filtered_data[filtered_data[STANDARD_COLUMNS['country']].isin(filters['countries'])]
    
    # Use st.session_state.date_range directly
    if 'date_range' in st.session_state and st.session_state.date_range:
        date_range = st.session_state.date_range
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = filtered_data[
                (pd.to_datetime(filtered_data[STANDARD_COLUMNS['date']]) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(filtered_data[STANDARD_COLUMNS['date']]) <= pd.to_datetime(end_date))
            ]
    
    return filtered_data

def calculate_performance_gaps(ts: pd.Series) -> pd.Series:
    """Calculate performance gaps as percentage deviation from rolling mean."""
    rolling_mean = ts.rolling(window=3, min_periods=1).mean()
    gaps = ((ts - rolling_mean) / rolling_mean * 100).fillna(0)
    return gaps


def guide_adjustment(forecast: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
    material_col = STANDARD_COLUMNS['material']
    demand_col = STANDARD_COLUMNS['demand']
    date_col = STANDARD_COLUMNS['date']
    
    logger.info(f"Historical data: {len(historical_data)} rows, columns: {list(historical_data.columns)}")
    logger.info(f"Forecast data: {len(forecast)} rows, columns: {list(forecast.columns)}")
    
    if historical_data.empty or forecast.empty:
        logger.error("Empty input data: historical_data or forecast is empty")
        return pd.DataFrame()
    
    # Ensure date column is datetime
    historical_data = historical_data.copy()
    historical_data[date_col] = pd.to_datetime(historical_data[date_col])
    
    patterns = {}
    for material in historical_data[material_col].unique():
        material_data = historical_data[historical_data[material_col] == material]
        material_data = material_data.sort_values(date_col)
        
        if len(material_data) < 4:
            logger.warning(f"Insufficient data for material {material}: {len(material_data)} rows")
            continue
        
        rolling_mean = material_data[demand_col].rolling(window=4, min_periods=1).mean()
        seasonal_factor = calculate_seasonal_factor(material_data, date_col, demand_col)
        
        adjusted_pattern = rolling_mean * seasonal_factor
        patterns[material] = adjusted_pattern.iloc[-1] if not adjusted_pattern.empty else 0
    
    logger.info(f"Patterns generated for {len(patterns)} materials")
    
    suggestions = []
    for _, row in forecast.iterrows():
        material = row.get(material_col, 'Unknown')
        if material not in patterns:
            logger.warning(f"Material {material} not in patterns, skipping")
            continue
        current_forecast = row.get('forecast', 0)
        suggested_demand = patterns.get(material, current_forecast)
        date = row.get(date_col, 'Unknown')
        
        confidence = calculate_confidence(historical_data, material, material_col, demand_col)
        
        suggestions.append({
            material_col: material,
            date_col: date,
            'current_forecast': current_forecast,
            'suggested_demand': suggested_demand,
            'adjustment_factor': (suggested_demand / current_forecast) if current_forecast > 0 else 1.0,
            'confidence': confidence,
            'reasoning': generate_reasoning(current_forecast, suggested_demand, confidence)
        })
    
    suggestions_df = pd.DataFrame(suggestions)
    logger.info(f"Suggestions DataFrame: {len(suggestions_df)} rows, unique materials: {suggestions_df[material_col].unique()}")
    return suggestions_df


def render_adjustment_wizard():
    """Render the step-by-step adjustment wizard with AI suggestions."""
    st.sidebar.markdown("### 🧙‍♂️ Adjustment Wizard")
    
    # Initialize wizard state
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    elif st.session_state.wizard_step not in [1, 2, 3]:
        logger.warning(f"Invalid wizard step: {st.session_state.wizard_step}. Resetting to 1.")
        st.session_state.wizard_step = 1
    
    # Step indicators
    steps = ["📊 Review Suggestions", "⚙️ Make Adjustments", "✅ Apply Changes"]
    current_step = st.session_state.wizard_step
    
    # Display step progress
    for i, step in enumerate(steps, 1):
        if i == current_step:
            st.sidebar.markdown(f"**{step}** ← Current")
        elif i < current_step:
            st.sidebar.markdown(f"~~{step}~~ ✓")
        else:
            st.sidebar.markdown(f"{step}")
    
    st.sidebar.markdown("---")
    
    # Step 1: Review Suggestions
    if current_step == 1:
        st.sidebar.markdown("**Step 1: Review AI Suggestions**")
        st.sidebar.info("💡 AI has analyzed historical patterns to suggest optimal demand forecasts.")
        
        if st.sidebar.button("View Suggestions", type="primary", key="view_suggestions_step1"):
            st.session_state.wizard_step = 2
            st.rerun()
    
    # Step 2: Make Adjustments
    elif current_step == 2:
        st.sidebar.markdown("**Step 2: Adjust Forecasts**")
        
        # Get current forecast data
        if "DeepAR" in st.session_state.state.forecasts:
            forecast_data = st.session_state.state.forecasts["DeepAR"]
            historical_data = st.session_state.state.data
            
            if historical_data is not None and not historical_data.empty:
                suggestions = guide_adjustment(forecast_data, historical_data)
                logger.info(f"Suggestions DataFrame: {len(suggestions)} rows, columns: {list(suggestions.columns)}")
                
                if suggestions.empty:
                    st.sidebar.error("No adjustment suggestions available. Check data.", icon="🚨")
                    return
                
                # Aggregate suggestions by material to avoid duplicate sliders
                """suggestions_agg = suggestions.groupby(STANDARD_COLUMNS['material']).agg({
                    'suggested_demand': 'mean',
                    'current_forecast': 'mean'
                }).reset_index()"""
                
                # Display adjustment sliders
                st.sidebar.markdown("**Adjust by Material and Period:**")
                
                adjustments = {}
                """for idx, row in suggestions_agg.head(5).iterrows():
                    material = row[STANDARD_COLUMNS['material']]
                    suggested = float(row['suggested_demand']) if pd.notna(row['suggested_demand']) else 100.0
                    
                    adjustment = st.sidebar.slider(
                        f"{material}",
                        min_value=0.0,
                        max_value=suggested * 2,
                        value=suggested,
                        step=1.0,
                        help=f"AI suggests: {suggested:.1f}",
                        key=f"slider_{material}_{idx}"  # Unique key
                    )
                    adjustments[material] = adjustment
                
                st.session_state.forecast_adjustments = adjustments"""

                for idx, row in suggestions.head(5).iterrows():
                    material = row[STANDARD_COLUMNS['material']]
                    date = row.get('date', f"Period_{idx}")
                    suggested = float(row['suggested_demand']) if pd.notna(row['suggested_demand']) else 100.0
                    
                    adjustment = st.sidebar.slider(
                        f"{material} ({date})",
                        min_value=0.0,
                        max_value=suggested * 2,
                        value=suggested,
                        step=1.0,
                        help=f"AI suggests: {suggested:.1f}",
                        key=f"slider_{material}_{date}_{idx}"
                    )
                    adjustments[f"{material}_{date}"] = adjustment

                st.session_state.forecast_adjustments = adjustments
                logger.info(f"Forecast adjustments set: {adjustments}")
                st.write(f"DEBUG: Adjustments set: {adjustments}")
        else:
            st.sidebar.error("No forecast data available. Generate a forecast first.", icon="🚨")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("← Back", key="back_step2"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("Next →", type="primary", key="next_step2"):
                if hasattr(st.session_state, 'forecast_adjustments') and st.session_state.forecast_adjustments:
                    st.session_state.wizard_step = 3
                    st.rerun()
                else:
                    st.sidebar.error("No adjustments made. Please adjust at least one material.", icon="🚨")
    
    # Step 3: Apply Changes
    elif current_step == 3:
        st.sidebar.markdown("**Step 3: Apply Adjustments**")
        st.sidebar.success("🎯 Ready to apply your forecast adjustments!")
        
        if st.sidebar.button("Apply Adjustments", type="primary", key="apply_adjustments_step3"):
            logger.info("Apply Adjustments button clicked")
            
            if hasattr(st.session_state, 'forecast_adjustments') and st.session_state.forecast_adjustments:
                try:
                    # Apply adjustments to the forecast
                    forecast_df = st.session_state.state.forecasts["DeepAR"].copy()
                    material_col = STANDARD_COLUMNS['material']
                    date_col = STANDARD_COLUMNS['date']
                    
                    if material_col not in forecast_df.columns or date_col not in forecast_df.columns:
                        st.sidebar.error("Material or date column missing in forecast data.", icon="🚨")
                        return
                    
                    # Apply each adjustment

                    """for material, adjusted_value in st.session_state.forecast_adjustments.items():
                        mask = forecast_df[material_col] == material
                        forecast_df.loc[mask, 'forecast'] = adjusted_value
                        forecast_df.loc[mask, 'lower_bound'] = adjusted_value * 0.9
                        forecast_df.loc[mask, 'upper_bound'] = adjusted_value * 1.1
                    
                    # Commit changes
                    st.session_state.state.forecasts["DeepAR"] = forecast_df
                    logger.info(f"Applied adjustments to {len(st.session_state.forecast_adjustments)} materials")"""

                    for key, adjusted_value in st.session_state.forecast_adjustments.items():
                    # Parse material and date from key
                        material, date = key.rsplit('_', 1)
                        try:
                            # Convert date string to datetime for matching
                            date = pd.to_datetime(date)
                        except ValueError:
                            # Fallback for non-datetime period identifiers
                            date = date
                        # Update forecast for matching material and date
                        mask = (forecast_df[material_col] == material) & (forecast_df[date_col].astype(str).str.contains(str(date)))
                        if mask.any():
                            forecast_df.loc[mask, 'forecast'] = adjusted_value
                            forecast_df.loc[mask, 'lower_bound'] = adjusted_value * 0.9
                            forecast_df.loc[mask, 'upper_bound'] = adjusted_value * 1.1
                        else:
                            logger.warning(f"No matching forecast row for material {material} and date {date}")
                    st.session_state.state.forecasts["DeepAR"] = forecast_df
                    logger.info(f"Updated forecast with adjustments: {st.session_state.forecast_adjustments}")
                    st.sidebar.success("✅ Adjustments applied successfully!")
                    
                except Exception as e:
                    logger.error(f"Adjustment failed: {str(e)}")
                    st.sidebar.error(f"Failed to apply adjustments: {str(e)}", icon="🚨")
            else:
                st.sidebar.warning("No adjustments to apply", icon="⚠️")
        
        if st.sidebar.button("← Back to Adjustments", key="back_step3"):
            # Clear any existing adjustments when going back
            if hasattr(st.session_state, 'forecast_adjustments'):
                del st.session_state.forecast_adjustments
            st.session_state.wizard_step = 2
            st.rerun()



def render_mobile_ui():
    """Render mobile-optimized interface using Bootstrap."""
    import streamlit.components.v1 as components
    
    # Detect if mobile (simplified detection)
    is_mobile = st.session_state.get('is_mobile', False)
    
    if st.sidebar.checkbox("📱 Mobile Mode", value=is_mobile):
        st.session_state.is_mobile = True
        
        components.html("""
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <div class="container-fluid">
            <div class="row">
                <div class="col-12">
                    <div class="card mb-3">
                        <div class="card-header">
                            <h5 class="card-title mb-0">📊 My Forecasts</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-6">
                                    <div class="btn-group-vertical d-grid gap-2" role="group">
                                        <button type="button" class="btn btn-primary btn-lg">View Forecasts</button>
                                        <button type="button" class="btn btn-outline-primary btn-lg">Quick Adjust</button>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="card bg-light">
                                        <div class="card-body text-center">
                                            <h6 class="card-title">Quick Stats</h6>
                                            <p class="card-text">📈 +15% vs last month</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h6 class="card-title mb-0">⚡ Quick Actions</h6>
                        </div>
                        <div class="card-body">
                            <div class="d-grid gap-2">
                                <button class="btn btn-success btn-lg" type="button">✅ Approve All</button>
                                <button class="btn btn-warning btn-lg" type="button">⚠️ Flag Issues</button>
                                <button class="btn btn-info btn-lg" type="button">💬 Send Feedback</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <style>
        .card { border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .btn-lg { padding: 12px 20px; font-size: 16px; }
        .card-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        </style>
        """, height=400)
    else:
        st.session_state.is_mobile = False


def collect_feedback(user_id, forecast_id, comment):
    """Collect user feedback for forecast improvements."""
    from cls_session_management import SessionState
    
    # Initialize feedback list if not exists
    if not hasattr(st.session_state.state, '_feedback'):
        st.session_state.state._feedback = []
    
    feedback_entry = {
        "user_id": user_id,
        "forecast_id": forecast_id,
        "comment": comment,
        "timestamp": pd.Timestamp.now(),
        "status": "pending"
    }
    
    st.session_state.state._feedback.append(feedback_entry)
    st.success("✅ Feedback submitted! Data scientists will review your input.")
    
    return feedback_entry


def render_feedback_widget():
    """Render floating feedback widget."""
    # Floating feedback button
    if st.button("💬", help="Send Feedback", key="feedback_btn"):
        st.session_state.show_feedback = True
    
    # Feedback modal/popup
    if st.session_state.get('show_feedback', False):
        with st.expander("📝 Send Feedback to Data Scientists", expanded=True):
            st.markdown("**Help us improve the forecasts!**")
            
            col1, col2 = st.columns(2)
            with col1:
                user_id = st.text_input("Your ID", value="planner_001")
                forecast_id = st.selectbox("Forecast Type", ["DeepAR", "XGBoost", "ARIMA"])
            
            with col2:
                feedback_type = st.selectbox("Feedback Type", [
                    "Forecast too high", 
                    "Forecast too low", 
                    "Missing seasonality", 
                    "Other"
                ])
            
            comment = st.text_area(
                "Comments", 
                placeholder="Describe the issue or suggestion...",
                help="Be specific about which SKUs or time periods are affected."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Feedback", type="primary"):
                    if comment.strip():
                        collect_feedback(user_id, forecast_id, f"{feedback_type}: {comment}")
                        st.session_state.show_feedback = False
                        st.rerun()
                    else:
                        st.error("Please provide a comment.")
            
            with col2:
                if st.button("Cancel"):
                    st.session_state.show_feedback = False
                    st.rerun()


def calculate_seasonal_factor(data: pd.DataFrame, date_col: str, demand_col: str) -> float:
    """Calculate seasonal adjustment factor based on historical patterns."""
    if len(data) < 12:  # Need at least a year of data
        return 1.0
    
    data = data.copy()
    data['month'] = pd.to_datetime(data[date_col]).dt.month
    current_month = pd.to_datetime(data[date_col]).iloc[-1].month
    
    # Calculate monthly averages
    monthly_avg = data.groupby('month')[demand_col].mean()
    overall_avg = data[demand_col].mean()
    
    if overall_avg == 0:
        return 1.0
    
    seasonal_factor = monthly_avg.get(current_month, overall_avg) / overall_avg
    return max(0.5, min(2.0, seasonal_factor))  # Cap between 0.5 and 2.0

def calculate_confidence(historical_data: pd.DataFrame, material: str, material_col: str, demand_col: str) -> float:
    """Calculate confidence score based on historical variance."""
    material_data = historical_data[historical_data[material_col] == material]
    
    if len(material_data) < 3:
        return 0.5  # Low confidence for insufficient data
    
    # Calculate coefficient of variation
    mean_demand = material_data[demand_col].mean()
    std_demand = material_data[demand_col].std()
    
    if mean_demand == 0:
        return 0.3
    
    cv = std_demand / mean_demand
    # Convert CV to confidence (lower variance = higher confidence)
    confidence = max(0.1, min(1.0, 1 / (1 + cv)))
    return confidence

def generate_reasoning(current: float, suggested: float, confidence: float) -> str:
    """Generate human-readable reasoning for the adjustment suggestion."""
    if abs(current - suggested) / max(current, 1) < 0.05:
        return "Current forecast aligns well with historical patterns"
    
    change_pct = ((suggested - current) / max(current, 1)) * 100
    confidence_text = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
    
    if change_pct > 10:
        return f"Suggest increasing by {change_pct:.1f}% based on historical trends ({confidence_text} confidence)"
    elif change_pct < -10:
        return f"Suggest decreasing by {abs(change_pct):.1f}% based on historical trends ({confidence_text} confidence)"
    else:
        return f"Minor adjustment of {change_pct:.1f}% suggested ({confidence_text} confidence)"

def detect_mobile_device() -> bool:
    """Detect if the user is on a mobile device."""
    # This is a simple heuristic - in a real app, you might use JavaScript
    # For now, we'll use session state to allow manual toggle
    return st.session_state.get('mobile_mode', False)