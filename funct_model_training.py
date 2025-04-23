import streamlit as st
from consts_model import MODEL_TYPES, DEFAULT_FORECAST_PERIOD
from sklearn.model_selection import train_test_split
from cls_forecast_engine import ForecastEngine
from cls_data_preprocessor import DataProcessor
from cls_model_trainer import ModelTrainer

from funct_plot_predictions import plot_predictions


def show_model_training():
    st.header("Model Training")

    if st.session_state.state.data is None:
        st.warning("No data loaded. Please configure and load data first.")
        return

    df = st.session_state.state.data
    ts = DataProcessor.prepare_time_series(df)

    if ts is None:
        st.error("Could not prepare time series data for modeling.")
        return

    model_type = st.selectbox("Select Model Type", list(MODEL_TYPES.keys()))
    forecast_period = st.number_input("Forecast Period (months)",
                                      min_value=1, max_value=24,
                                      value=DEFAULT_FORECAST_PERIOD,
                                        key="training_forecast_period_input")
    test_size = st.slider("Test Size Percentage", 10, 40, 20)

    if st.button("Train Model"):
        with st.spinner("Training model..."):

            if model_type == "XGBoost":
                features_df = ForecastEngine.create_features(ts.to_frame(), ts.name)
                X = features_df.drop(columns=[ts.name, 'date'])
                y = features_df[ts.name]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, shuffle=False)

                model = ModelTrainer.train_xgboost(X_train, y_train)
                evaluation = ModelTrainer.evaluate_model(model, X_test, y_test)

                last_values = X.iloc[-1].values

                st.session_state.state.models[model_type] = {
                    'model': model,
                    'last_values': last_values,
                    'evaluation': evaluation
                }

                actual_index = y_test.index
                actual_values = y_test.values

            elif model_type == "ARIMA":
                train_size = int(len(ts) * (1 - test_size/100))
                train, test = ts[:train_size], ts[train_size:]

                model = ModelTrainer.train_arima(train)
                evaluation = ModelTrainer.evaluate_model(model, test.index, test.values)

                st.session_state.state.models[model_type] = {
                    'model': model,
                    'evaluation': evaluation
                }

                actual_index = test.index
                actual_values = test.values

            # Show results
            st.success(f"{model_type} model trained successfully!")
            st.subheader("Model Evaluation")
            st.write(f"Mean Absolute Error (MAE): {evaluation['mae']:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {evaluation['rmse']:.2f}")

            plot_predictions(actual_index, actual_values, evaluation['predictions'])