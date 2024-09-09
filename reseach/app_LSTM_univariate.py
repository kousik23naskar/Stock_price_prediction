import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from io import BytesIO

# Function to fetch historical stock price data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Preprocess data
def preprocess_data(data):
    prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    return scaled_prices, scaler

# Create dataset with look_back window
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# Define the model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape)) # input_shape=(n_input,n_features)
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=[RootMeanSquaredError()])
    print(model.summary())
    return model

# Train and save model
def train_and_save_model(X_train, y_train, model_path):
    model = build_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, batch_size=32, callbacks=[early_stopping], verbose=1)
    model.save(model_path)
    return model, history

# Predict future prices
def predict_future_prices(model, scaler, last_known_data, future_days, look_back):
    scaled_last_known_data = scaler.transform(last_known_data)
    X_input = scaled_last_known_data[-look_back:].reshape((1, look_back, 1))
    
    future_prices = []
    for _ in range(future_days):
        prediction = model.predict(X_input)
        
        # Reshape prediction to match dimensions of X_input
        prediction_reshaped = prediction.reshape((1, 1, 1))
        
        # Update X_input by appending the new prediction
        X_input = np.append(X_input[:, 1:, :], prediction_reshaped, axis=1)
        
        # Append the new prediction to the future_prices list
        future_prices.append(prediction[0, 0])
    
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices

# Streamlit app
def main():
    st.title('Stock Price Forecasting App')
    
    ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2024-08-01'))
    future_days = st.sidebar.slider('Days to Predict', min_value=1, max_value=365, value=180)
    
    model_path = 'stock_price_model_univariate.h5'
    
    if os.path.exists(model_path):
        st.sidebar.text('Model found. You can use it for predictions.')
    else:
        st.sidebar.text('Model not found. Please train the model.')
    
    if st.sidebar.button('Train Model'):
        with st.spinner('Training model...'):
            data = fetch_data(ticker, start_date, end_date)
            scaled_data, scaler = preprocess_data(data)
            look_back = 60
            X, y = create_dataset(scaled_data, look_back)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            
            model, history = train_and_save_model(X_train, y_train, model_path)
            st.sidebar.text('Model trained and saved!')
            st.success('Model training complete!')
            
            # Plot training and validation loss
            st.subheader('Training and Validation Loss')
            plt.figure(figsize=(14, 7))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            # Save the plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            st.image(buf.getvalue())
            buf.close()
    else:
        st.sidebar.text('Click on "Train Model"')
    
    if os.path.exists(model_path):
        data = fetch_data(ticker, start_date, end_date)
        scaled_data, scaler = preprocess_data(data)
        look_back = 60
        last_known_data = data[['Close']].values[-look_back:]
        
        if st.button('Forecast Future Prices'):
            with st.spinner('Forecasting future prices...'):
                model = load_model(model_path)
                future_prices = predict_future_prices(model, scaler, last_known_data, future_days, look_back)
                
                future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
                
                # Display forecast results
                forecast_df = pd.DataFrame(data={'Date': future_dates, 'Predicted Price': future_prices.flatten()})
                st.write(forecast_df)
                
                # Plot results
                plt.figure(figsize=(14, 7))
                plt.plot(data.index, data['Close'], label='Historical Prices')
                plt.plot(future_dates, future_prices, label='Forecasted Prices', color='red')
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