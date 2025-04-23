import pandas as pd
import numpy as np


# Forecasting Module
class ForecastEngine:
    @staticmethod
    def create_features(df, target_col, lags=3):
        """Create time series features for supervised learning"""
        df = df.copy()
        df['date'] = df.index
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        
        # Create lag features
        for lag in range(1, lags + 1):
            df[f'lag_{lag}'] = df[target_col].shift(lag)
            
        df.dropna(inplace=True)
        return df

    @staticmethod
    def forecast_xgboost(model, last_known_values, periods, last_date, freq='M'):
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq=freq
        )
        
        forecast = []
        current_features = last_known_values.copy()

        for _ in range(periods):
            next_pred = model.predict(current_features.reshape(1, -1))[0]
            forecast.append(next_pred)

            current_features = np.roll(current_features, -1)
            current_features[-1] = next_pred

        return pd.Series(forecast, index=future_dates)


    @staticmethod
    def forecast_arima(model, periods):
        """Generate forecasts using ARIMA model"""
        forecast = model.forecast(steps=periods)
        return forecast