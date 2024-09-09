import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
import streamlit as st
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt
import io
import os

#**2. Feature Engineering**
def add_features(stock_data, sma_window=20, rsi_window=14):
    stock_data = stock_data.copy()
    stock_data['Price Percentage Change'] = stock_data['Adj Close'].pct_change(periods=7) * 100
    stock_data['SMA'] = stock_data['Adj Close'].rolling(window=sma_window).mean()
    stock_data['RSI'] = RSIIndicator(stock_data['Adj Close'], window=rsi_window).rsi()
    stock_data.dropna(inplace=True)
    return stock_data

#**3. Load Data**
def load_data(ticker, start_date='2010-01-01', end_date='2024-07-01'):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    #data = add_features(data)
    return data

#**4. Preprocessing and Model Training**
def preprocess_data(data, feature_cols, target_cols, sequence_length):
    # Feature and target data
    features = data[feature_cols]
    target = data[target_cols]
    
    # Split the data
    split_index = int(len(features) * 0.8)
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = target[:split_index], target[split_index:]
    
    # Normalize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Create sequences
    def create_sequences(data, target, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(target[i + window_size])
        return np.array(X), np.array(y)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
    
    return (X_train_seq, y_train_seq), (X_test_seq, y_test_seq), scaler_X, scaler_y

def build_model(input_shape, output_dim, dropout_rate=0.3):
    model = Sequential()
    model.add(Input(shape=input_shape)) # input_shape=(n_input,n_features)
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(dropout_rate))
    #model.add(Dense(50, activation='relu')),
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    return model

#**5. Train and Save Model**
def train_and_save_model(X_train, y_train, output_dim, model_path='lstm_model.h5'):
    model = build_model((X_train.shape[1], X_train.shape[2]), output_dim, dropout_rate=0.3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    model.save(model_path)
    return model

#**6. Prediction Function**
def predict_future_prices(model, scaler_X, scaler_y, data, feature_cols, sequence_length, target_cols, future_days):
    # Prepare the last sequence for prediction
    last_sequence = data[feature_cols].values[-sequence_length:]
    last_sequence_scaled = scaler_X.transform(last_sequence)
    last_sequence_scaled = np.expand_dims(last_sequence_scaled, axis=0)
    
    # Predict future values
    future_predictions = []
    for _ in range(future_days):
        # Predict the next value
        prediction_scaled = model.predict(last_sequence_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        future_predictions.append(prediction[0])
        
        # Update the sequence: remove the first step and append the new prediction
        last_sequence_scaled = np.roll(last_sequence_scaled, shift=-1, axis=1)
        last_sequence_scaled[0, -1, :] = scaler_X.transform(
            pd.DataFrame([prediction[0]], columns=feature_cols).values
        ).flatten()
    
    return np.array(future_predictions)


#**7. Streamlit Application**
def main():
    st.title("Stock Price Prediction")

    # User inputs
    ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2024-07-01'))
    future_days = st.sidebar.slider('Days to Predict', min_value=1, max_value=365, value=90)
    
    data = load_data(ticker, start_date=start_date, end_date=end_date)
    
    feature_cols = ['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']
    #feature_cols = ['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume', 'Price Percentage Change', 'SMA', 'RSI']
    target_cols = ['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']
    
    sequence_length = 60
    (X_train, y_train), (X_test, y_test), scaler_X, scaler_y = preprocess_data(data, feature_cols, target_cols, sequence_length)
    
    output_dim = len(target_cols)
    
    model_path = 'lstm_model.h5'
    
    if not os.path.exists(model_path):
        st.sidebar.text('Model not found.\nPlease train the model.')
        
        if st.sidebar.button('Train Model'):
            with st.spinner('Training model...'):
                model = train_and_save_model(X_train, y_train, output_dim, model_path)
                st.sidebar.text('Model trained and saved!')
                st.success('Model training complete!')
        else:
            st.sidebar.text('Click on "Train Model"')
            return  # Exit early if model is not available and the user has not trained it
    else:
        model = load_model(model_path)
    
    # Predict
    predictions = predict_future_prices(model, scaler_X, scaler_y, data, feature_cols, sequence_length, target_cols, future_days)
    predictions_adj_close = predictions[:, target_cols.index('Adj Close')]
    
    # Plot and display predictions
    fig, ax = plt.subplots()
    
    # Create prediction dates manually
    last_date = data.index[-1]
    prediction_dates = [last_date + pd.DateOffset(days=i) for i in range(1, future_days + 1)]
    
    ax.plot(data.index, data['Adj Close'], label='Historical Prices')
    ax.plot(prediction_dates, predictions_adj_close, label='Predicted Prices', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{ticker} Price Prediction')
    ax.legend()
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()