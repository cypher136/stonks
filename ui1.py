import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt

# Title
st.title("STONKS")

# Taking user inputs for stock name and prediction duration
stock = st.text_input("Enter stock name to predict prices:")
duration = st.number_input("Enter number of days to predict stock prices:", min_value=1)

# Define the sequence length for data preparation
sequence_length = 100

if st.button("Predict"):
    # Downloading historical stock data using yfinance
    df = yf.download(stock, start="2010-01-01", end=None)
    df = df.reset_index()

    # Splitting data into training and testing sets
    data_training = pd.DataFrame(df["Close"][:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70): int(len(df))])

    # Applying MinMax scaling to training data
    scaler = MinMaxScaler()
    data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

    # Creating training sequences and targets
    x_train = []
    y_train = []
    for i in range(data_training_array.shape[0] - 100):
        x_train.append(data_training_array[i: 100 + i, 0])
        y_train.append(data_training_array[100 + i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Creating and compiling an LSTM model using Keras
    model = keras.Sequential([
        keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(60, activation='relu', return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(80, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=30, batch_size=30)

    # Preparing data for future predictions
    past_days = data_training.tail(100)
    new_df = past_days._append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(new_df)

    # Creating test sequences and targets for future predictions
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Making predictions on test data
    y_predicted = model.predict(x_test)

    # Scaling predictions back to original values
    scale_factor = 1 / 0.02099617
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Creating an array of input data for future predictions
    x_future = [input_data[-100:]]

    # Predicting future stock prices
    future_predictions = []
    for i in range(duration):
        x_future_array = np.array(x_future)
        x_future_array = np.reshape(x_future_array, (x_future_array.shape[0], x_future_array.shape[1], 1))
        future_prediction = model.predict(x_future_array)
        future_predictions.append(future_prediction[0, 0])
        x_future[0] = np.concatenate((x_future[0][1:], future_prediction), axis=None)

    # Inverse scaling and transforming future predictions
    future_predictions_array = np.array(future_predictions)
    y_future = scaler.inverse_transform(future_predictions_array.reshape(-1, 1))

    # Creating time indices for future predictions
    future_time_indices = pd.date_range(start=df["Date"].iloc[-1], periods=duration, freq="D")

    # Displaying and plotting the predicted future stock prices
    st.write("Predicted Future Stock Prices:")
    st.write(y_future)

    fig, ax = plt.subplots()
    ax.plot(df["Date"][-100:], df["Close"][-100:], label='Historical', color='b')
    ax.plot(future_time_indices, y_future, label='Predicted', color='r')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)