import numpy as np
import pandas as pd
from datetime import timedelta
import joblib
import pickle
import matplotlib.pyplot as plt

class StockForecasting:
    def import_data(self, file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    def generate_future_predictions(self, data, ticker, model_path, scaler_path, days_to_predict):
        # Load ARIMA model and scaler
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        scaler = joblib.load(scaler_path)
        
        # Get recent data for scaling
        recent_data = data[[ticker]].values[-days_to_predict:]
        recent_data_scaled = scaler.transform(recent_data)
        
        # Define start and end dates for forecast
        start = data.index[-1]
        end = start + timedelta(days=days_to_predict)
        
        # Generate forecast
        forecast_scaled = model.forecast(steps=days_to_predict)
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()  # Scale back to original values
        
        # Create date range for predictions
        date_range = pd.date_range(start=start + timedelta(days=1), periods=days_to_predict)
        
        # Create DataFrame for forecast
        forecast_df = pd.DataFrame({ticker: forecast}, index=date_range)
        return forecast_df

    def visualize_forecast(self, ticker, historical_data, forecast_data, conf_interval=0.05): 
        conf_int_pct = 100 - conf_interval * 100
        
        # Confidence Intervals
        lower_bound = forecast_data[ticker].values * (1 - conf_interval)
        upper_bound = forecast_data[ticker].values * (1 + conf_interval)

        # Plotting
        plt.figure(figsize=(14, 8))
        plt.plot(historical_data.index, historical_data[ticker], color='blue', label=f'{ticker} Historical Data')
        plt.plot(forecast_data.index, forecast_data[ticker], color='orange', label=f'{ticker} Forecasted Price')
        plt.fill_between(forecast_data.index, lower_bound, upper_bound, color='gray', alpha=0.3, label='Confidence Interval')
        
        plt.title(f"{ticker} Stock Price Forecast with {conf_int_pct}% Confidence Interval")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
        
    def evaluate_forecast(self, ticker, forecast_data, conf_interval=0.05):
        # Trend Analysis
        direction = "increasing" if forecast_data[ticker].iloc[-1] > forecast_data[ticker].iloc[0] else "decreasing"
        print(f"Trend Evaluation: The forecast indicates a `{direction}` trend.")
        variance = forecast_data[ticker].var()
        volatility = np.sqrt(variance * 252)
        
        # Volatility and Risk Analysis
        print(f"Volatility Evaluation: The forecasted data exhibits a volatility level of ${volatility:.2f}$.")
        
        # Market Insights
        if direction == "increasing":
            print("Growth Potential: A positive trend suggests possible price increases.")
        else:
            print("Risk Indicator: The downward trend suggests potential risks with price declines.")
    
    def combine_and_export(self, data1, data2, data3):
        # Concatenate along the Date index, keeping only rows with matching indices
        combined_df = pd.concat([data1, data2, data3], axis=1, join='inner').reset_index()
        combined_df.to_csv("../data/processed/forecast_combined.csv")
