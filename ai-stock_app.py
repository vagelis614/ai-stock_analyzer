import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

# --- App title ---
st.title("📈 AI Penny Stock Analyzer")

# --- Input ---
symbol = st.text_input("Enter stock ticker (e.g., SNDL, NAKD, CEI)", value="CEI")

# --- Load Data ---
@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, period="6mo", interval="1d")
    return data

data = load_data(symbol)

# --- Handle missing or empty data ---
if data.empty or data.isnull().values.any():
    st.error("No valid data found for the given symbol.")
    st.stop()

data.dropna(inplace=True)

# --- Convert OHLC columns to 1D Series for TA-lib compatibility ---
close_prices = pd.Series(data['Close'].values.flatten(), index=data.index)
high_prices = pd.Series(data['High'].values.flatten(), index=data.index)
low_prices = pd.Series(data['Low'].values.flatten(), index=data.index)

# --- Indicators ---
data['RSI'] = ta.momentum.RSIIndicator(close=close_prices).rsi()

macd = ta.trend.MACD(close=close_prices)
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()
data['MACD_hist'] = macd.macd_diff()

adx = ta.trend.ADXIndicator(high=high_prices, low=low_prices, close=close_prices)
data['ADX'] = adx.adx()

# --- Plot Price and Volume ---
st.subheader("📊 Price & Volume")
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(data.index, data['Close'], label="Close Price", linewidth=2)
ax1.set_ylabel("Price")
ax2 = ax1.twinx()
ax2.bar(data.index, data['Volume'], alpha=0.2, label="Volume", color='gray')
ax2.set_ylabel("Volume")
fig.legend(loc="upper left")
st.pyplot(fig)

# --- Plot RSI ---
st.subheader("💡 RSI")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(data.index, data['RSI'], label="RSI", color='purple')
ax.axhline(70, linestyle='--', color='red')
ax.axhline(30, linestyle='--', color='green')
ax.set_ylim([0, 100])
ax.set_ylabel("RSI")
st.pyplot(fig)

# --- Plot MACD ---
st.subheader("📉 MACD")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(data.index, data['MACD'], label="MACD", color='blue')
ax.plot(data.index, data['MACD_signal'], label="Signal", color='orange')
ax.bar(data.index, data['MACD_hist'], label="Histogram", alpha=0.3)
ax.set_ylabel("MACD")
ax.legend()
st.pyplot(fig)

# --- Plot ADX ---
st.subheader("📌 ADX")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(data.index, data['ADX'], label="ADX", color='black')
ax.axhline(25, linestyle='--', color='gray')
ax.set_ylabel("ADX Strength")
st.pyplot(fig)
