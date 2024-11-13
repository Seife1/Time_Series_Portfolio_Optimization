import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import joblib

import itertools
import warnings
warnings.filterwarnings('ignore')
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s') 


class ARIMAModel:
    def load_dataset(self, file_path):
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        return data
    
    def assess_stationarity(self, data, column):
        result = adfuller(data[column], autolag="AIC")
        
        print(f"ADF Test Statistic: {result[0]}")
        print(f"p-Value: {result[1]}")
        print(f"Lag Used: {result[2]}")
        print(f"Observations Used: {result[3]}")
        print(f"Critical Values: {result[4]}")
        print(f"Stationarity Conclusion: {'Stationary' if result[1] < 0.05 else 'Non-Stationary'}")
        
        logging.info("Checked time series stationarity status.")
        
    def split_train_test(self, data, column):
        train_data = data[[column]]
        train_size = int(len(train_data) * 0.8)
        train, test = train_data[column][:train_size], train_data[column][train_size:]
        logging.info("Split dataset into training and testing sets (80/20 ratio).")
        
        return train, test
    

    def optimize_arima_order(self, train):
        # Grid search over p, d, q parameters
        p_values = range(0, 5)
        d_values = range(0, 3)
        q_values = range(0, 5)

        best_aic = float('inf')
        best_order = None
        best_model = None

        for p, d, q in itertools.product(p_values, d_values, q_values):
            try:
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except:
                continue

        print(f"Optimal ARIMA parameters found: {best_order}")
        logging.info("Performed grid search to identify the optimal ARIMA parameters.")
        
        return best_order
        
        
    def fit_arima_model(self, train, test, order, ticker):
        # Initialize and fit the scaler on the training data
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
        
        # Fit the ARIMA model on the scaled training data
        arima_model = ARIMA(scaled_train, order=order)
        model_fit = arima_model.fit()
        
        # Ensure the directory exists
        os.makedirs('../models', exist_ok=True)
        
        # Save both the ARIMA model and the scaler
        model_fit.save(f'../models/arima_model_{ticker}.pkl')
        joblib.dump(scaler, f'../models/scaler_{ticker}.joblib')
        
        # Forecast on scaled data, then inverse transform the forecast to original scale
        forecast_scaled = model_fit.forecast(steps=len(test))
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))
        
        logging.info("Trained ARIMA model and saved both model and scaler.")
        
        return forecast

    def evaluate_model_performance(self, actual, predicted):
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100  # MAPE as a percentage

        # Print metrics
        print("Evaluation Metrics:")
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")
        print(f"MAPE: {mape}%")
        
        logging.info("Calculated evaluation metrics for model performance.")

    def display_forecast_results(self, column, train, test, forecast):
        # Plot the final forecast
        plt.figure(figsize=(14, 7))
        plt.plot(train, label='Training Data')
        plt.plot(test, label='Test Data')
        plt.plot(test.index, forecast, label='ARIMA Prediction')
        plt.title(f"{column} - Actual vs Predicted")
        plt.legend()
        plt.show()
        
        logging.info("Displayed forecast comparison plot.")
