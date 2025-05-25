#renders_for_simple_mode_ui.py
import streamlit as st
import streamlit.components.v1 as components
def render_mobile_ui():
    """Render mobile-optimized interface."""
    components.html("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
    .mobile-card {
        margin-bottom: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .mobile-btn {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .swipe-container {
        overflow-x: auto;
        white-space: nowrap;
        padding: 10px 0;
    }
    .swipe-item {
        display: inline-block;
        width: 280px;
        margin-right: 15px;
        vertical-align: top;
        white-space: normal;
    }
    </style>
    <div class="container-fluid">
        <div class="row">
            <div class="col-12">
                <h3 class="text-center mb-4">📱 My Forecasts</h3>
                <div class="swipe-container">
                    <div class="swipe-item">
                        <div class="card mobile-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Quick Forecast</h5>
                                <p class="card-text">Generate instant demand forecast</p>
                                <button class="btn btn-primary mobile-btn" onclick="parent.window.postMessage({type: 'action', action: 'generate_forecast'}, '*')">Generate</button>
                            </div>
                        </div>
                    </div>
                    <div class="swipe-item">
                        <div class="card mobile-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Adjust Forecast</h5>
                                <p class="card-text">Fine-tune predictions</p>
                                <button class="btn btn-warning mobile-btn" onclick="parent.window.postMessage({type: 'action', action: 'adjust_forecast'}, '*')">Adjust</button>
                            </div>
                        </div>
                    </div>
                    <div class="swipe-item">
                        <div class="card mobile-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">View KPIs</h5>
                                <p class="card-text">Check performance metrics</p>
                                <button class="btn btn-info mobile-btn" onclick="parent.window.postMessage({type: 'action', action: 'view_kpis'}, '*')">View</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'action') {
            // Handle mobile actions
            console.log('Mobile action:', event.data.action);
        }
    });
    </script>
    """, height=300)

def detect_mobile():
    """Detect if user is on mobile device."""
    # Simple mobile detection using viewport width
    mobile_js = """
    <script>
    function isMobile() {
        return window.innerWidth <= 768;
    }
    parent.window.postMessage({type: 'mobile', isMobile: isMobile()}, '*');
    </script>
    """
    components.html(mobile_js, height=0)
    return st.session_state.get('is_mobile', False)

def render_mobile_controls():
    """Render mobile-specific controls."""
    st.markdown("### 📊 Quick Controls")
    
    # Mobile-friendly filters
    with st.expander("🔍 Filters", expanded=False):
        render_filters() # render_filters is still used here for mobile
    
    # Mobile forecast section
    with st.expander("📈 Forecast", expanded=True):
        render_forecast_section()