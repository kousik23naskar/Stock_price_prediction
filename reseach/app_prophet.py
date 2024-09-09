import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from prophet import Prophet
from datetime import datetime, timedelta
from io import BytesIO

# Function to fetch historical stock price data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Prepare data for Prophet
def prepare_prophet_data(data):
    prophet_data = data[['Close']].reset_index()
    prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'})
    return prophet_data

# Train and forecast using Prophet
def train_and_forecast_prophet(prophet_data, forecast_periods):
    train_size = int(len(prophet_data) * 0.8)
    prophet_train_data = prophet_data[:train_size]
    
    model_prophet = Prophet()
    model_prophet.fit(prophet_train_data)
    
    future = model_prophet.make_future_dataframe(periods=forecast_periods)
    forecast = model_prophet.predict(future)
    
    return model_prophet, forecast

# Streamlit app
def main():
    st.title('Stock Price Forecasting App')
    
    ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2024-08-01'))
    future_days = st.sidebar.slider('Days to Predict', min_value=1, max_value=365, value=180)
    
    model_path = 'stock_price_model_prophet.pkl'
    
    data = fetch_data(ticker, start_date, end_date)
    prophet_data = prepare_prophet_data(data)
    
    if st.sidebar.button('Train and Forecast with Prophet'):
        with st.spinner('Training Prophet model...'):
            model_prophet, forecast = train_and_forecast_prophet(prophet_data, future_days)
            
            # Plot training and forecast results
            st.subheader('Forecast Results')
            fig1 = model_prophet.plot(forecast)
            st.write(fig1)
            
            fig2 = model_prophet.plot_components(forecast)
            st.write(fig2)
            
            # Evaluate Prophet model
            train_size = int(len(prophet_data) * 0.8)
            prophet_test_data = prophet_data[train_size:]
            forecast_test = forecast.tail(len(prophet_test_data))
            
            mse_pr = mean_squared_error(prophet_test_data['y'], forecast_test['yhat'])
            mae_pr = mean_absolute_error(prophet_test_data['y'], forecast_test['yhat'])
            mape_pr = mean_absolute_percentage_error(prophet_test_data['y'], forecast_test['yhat'])
            
            st.write(f"Prophet Model - MSE: {mse_pr:.4f}, MAE: {mae_pr:.4f}, MAPE: {mape_pr:.4f}")
    else:
        st.sidebar.text('Click on "Train and Forecast with Prophet"')
    
    if st.button('Display Forecast Plot'):
        with st.spinner('Generating forecast plot...'):
            model_prophet, forecast = train_and_forecast_prophet(prophet_data, future_days)
            
            # Plot results
            plt.figure(figsize=(14, 7))
            plt.plot(data.index, data['Close'], label='Historical Prices')
            future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
            forecast_dates = forecast['ds'][-future_days:]
            plt.plot(forecast_dates, forecast['yhat'][-future_days:], label='Forecasted Prices', color='red')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Stock Price Forecast for {ticker}')
            plt.legend()
            
            # Save the plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            st.image(buf.getvalue())
            buf.close()

if __name__ == "__main__":
    main()
