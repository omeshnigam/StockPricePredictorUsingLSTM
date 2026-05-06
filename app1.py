import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime, timedelta

# Load the trained model
model = load_model(r'Stock Prediction using Model.keras')

st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'TSLA')

# Calculate date range for data
end = datetime.now().date()
start = end - timedelta(days=365*10)

# Download stock data
data = yf.download(stock, start=start, end=end)

st.subheader("Stock Data")
st.write(data)

# Prepare training and test datasets
data_train = pd.DataFrame(data.Close[:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# Initialize scaler and fit only on training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train.values.reshape(-1, 1))

# Prepare test data with 100 days overlap
pas_100_days = data_train.tail(100)
data_test_combined = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test_combined.values.reshape(-1, 1))

# Plot moving averages with legend
st.subheader("Price vs MA50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader("Price vs MA50 vs MA100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader("Price vs MA100 vs MA200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.show()
st.pyplot(fig3)

# Prepare sequences for prediction
x = []
y = []
for i in range(100, len(data_test_scaled)):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])
x, y = np.array(x), np.array(y)

# Model prediction
predict = model.predict(x)

# Inverse scaling
predict_original = scaler.inverse_transform(predict)
y_original = scaler.inverse_transform(y.reshape(-1, 1))

# Plot actual vs predicted prices
st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict_original, 'r', label='Predicted Price')
plt.plot(y_original, 'g', label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

# Display latest predicted price (for today)
predicted_price_today = predict_original[-1][0]
st.subheader(f"Predicted stock price for today is: ${predicted_price_today:.2f}")

