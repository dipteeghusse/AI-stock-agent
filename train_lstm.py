import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# 1. Load Data
data = yf.download("AAPL", start="2018-01-01", end="2024-01-01")
close_prices = data[['Close']].values

# 2. Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

# 3. Create sequences
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# 4. Build Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60,1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 5. Train
model.fit(X, y, epochs=10, batch_size=32)

# 6. Save Model + Scaler
model.save("model/lstm_model.h5")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model and scaler saved!")
