import plotly.express as px
import streamlit as st

def plot_predictions(actual_index, actual_values, predicted_values):
    fig = px.line(x=actual_index, y=actual_values, title="Actual vs Predicted")
    fig.add_scatter(x=actual_index, y=predicted_values, name='Predicted')
    st.plotly_chart(fig)