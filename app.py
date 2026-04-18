# -------------------------
# IMPORTS
# -------------------------
import streamlit as st
import numpy as np
import yfinance as yf
import joblib
import pandas as pd
import time
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Stock AI Dashboard", layout="wide")
st.title("📈 LSTM Stock Market Prediction Dashboard")

# -------------------------
# LOAD MODEL & SCALER
# -------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("model/lstm_model.h5", compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -------------------------
# STOCK SELECTION
# -------------------------
stock = st.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "GOOG", "TCS.NS", "RELIANCE.NS"]
)

# -------------------------
# DATA FETCH (TILL TODAY)
# -------------------------
today = datetime.today().strftime('%Y-%m-%d')

data = yf.download(stock, start="2018-01-01", end=today, progress=False)

# Fix multi-index issue
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

features = ["Open", "High", "Low", "Close", "Volume"]
df = data[features].dropna()

if df.empty or len(df) < 60:
    st.error("Not enough data")
    st.stop()

# -------------------------
# STOCK LINE CHART
# -------------------------
st.subheader("📊 Closing Price Trend")
st.line_chart(df["Close"])

# -------------------------
# LAST 10 DAYS DATA
# -------------------------
st.subheader("📅 Last 10 Days Data")

last10 = df.tail(10).reset_index()
last10["Date"] = pd.to_datetime(last10["Date"]).dt.strftime("%d-%b-%Y")
st.dataframe(last10)

# -------------------------
# CURRENT PRICE (SAFE)
# -------------------------
st.subheader("💰 Current Price")

current_price = df["Close"].iloc[-1]
if isinstance(current_price, (pd.Series, np.ndarray)):
    current_price = np.array(current_price).reshape(-1)[0]

current_price = float(current_price)

st.metric("Latest Close Price", f"{current_price:.2f}")

# -------------------------
# BUY / SELL SIGNALS (MA CROSSOVER)
# -------------------------
signal_df = df.copy()

signal_df["MA_10"] = signal_df["Close"].rolling(10).mean()
signal_df["MA_20"] = signal_df["Close"].rolling(20).mean()
signal_df.dropna(inplace=True)

signal_df["Signal"] = 0
signal_df.loc[signal_df["MA_10"] > signal_df["MA_20"], "Signal"] = 1
signal_df.loc[signal_df["MA_10"] < signal_df["MA_20"], "Signal"] = -1

buy = signal_df[signal_df["Signal"] == 1]
sell = signal_df[signal_df["Signal"] == -1]

# -------------------------
# CANDLESTICK CHART
# -------------------------
st.subheader("🕯️ Candlestick Chart with BUY/SELL Signals")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.tail(100).index,
    open=df["Open"].tail(100),
    high=df["High"].tail(100),
    low=df["Low"].tail(100),
    close=df["Close"].tail(100),
    name="Candlestick"
))

# BUY markers
fig.add_trace(go.Scatter(
    x=buy.index,
    y=buy["Close"],
    mode="markers",
    marker=dict(color="green", size=10, symbol="triangle-up"),
    name="BUY Signal"
))

# SELL markers
fig.add_trace(go.Scatter(
    x=sell.index,
    y=sell["Close"],
    mode="markers",
    marker=dict(color="red", size=10, symbol="triangle-down"),
    name="SELL Signal"
))

fig.update_layout(
    title=f"{stock} Trading Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# PREPARE INPUT FOR LSTM
# -------------------------
last60_multi = df.tail(60)
last60_single = df["Close"].tail(60).values.reshape(-1, 1)

num_features = scaler.n_features_in_

def get_scalar(pred):
    return float(np.array(pred).reshape(-1)[0])

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_days(days):
    predictions = []

    if num_features == 1:
        temp = last60_single.copy()

        for _ in range(days):
            scaled = scaler.transform(temp).reshape(1, 60, 1)
            pred = model.predict(scaled)
            val = get_scalar(pred)

            inv = scaler.inverse_transform([[val]])
            close = float(inv[0][0])

            predictions.append(close)
            temp = np.append(temp[1:], [[close]], axis=0)

    else:
        temp = last60_multi.values.copy()

        for _ in range(days):
            scaled = scaler.transform(temp).reshape(1, 60, num_features)
            pred = model.predict(scaled)
            val = get_scalar(pred)

            new_row = temp[-1].copy()
            new_row[3] = val

            dummy = np.zeros((1, num_features))
            dummy[0] = new_row
            inv = scaler.inverse_transform(dummy)

            close = float(inv[0][3])
            predictions.append(close)

            temp = np.vstack([temp[1:], new_row])

    return predictions

# -------------------------
# PREDICTION SETTINGS
# -------------------------
st.subheader("⚙️ Forecast Settings")

days = st.slider("Select prediction days", 3, 14, 7)
run = st.button("🚀 Generate Forecast")

# -------------------------
# EXECUTION
# -------------------------
if run:

    preds = predict_days(days)

    last_date = df.index[-1]
    dates = [
        (last_date + timedelta(days=i)).strftime("%d-%b-%Y")
        for i in range(1, days + 1)
    ]

    st.subheader(f"📅 Next {days} Days Prediction")

    for d, p in zip(dates, preds):
        st.write(f"{d} → ₹ {p:.2f}")

    st.line_chart(pd.DataFrame({
        "Date": dates,
        "Predicted": preds
    }).set_index("Date"))

    # -------------------------
    # INVESTMENT SUGGESTION
    # -------------------------
    st.subheader("📊 Investment Suggestion")

    avg_pred = np.mean(preds)
    change = ((avg_pred - current_price) / current_price) * 100

    st.write(f"📈 Expected Change: {change:.2f}%")

    if change > 2:
        st.success("🟢 STRONG BUY")
    elif change > 0.5:
        st.info("🟡 BUY / ACCUMULATE")
    elif change > -0.5:
        st.warning("🟠 HOLD")
    elif change > -2:
        st.error("🔴 WEAK SELL")
    else:
        st.error("🔴 STRONG SELL")

# -------------------------
# AUTO REFRESH
# -------------------------
time.sleep(60)
st.rerun()