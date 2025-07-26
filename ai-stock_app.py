import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

st.title("📈 AI Stock Analyzer")
symbol = st.text_input("📊 Enter Stock Ticker", "AAPL").upper()

if not symbol:
    st.warning("Please enter a stock symbol.")
    st.stop()

# Download stock data
data = yf.download(symbol, period="6mo", interval="1d")

if data.empty or data['Close'].isnull().all():
    st.error("No valid data found for the given symbol.")
    st.stop()

# Make sure Close is a Series (1D)
close = data['Close']
if isinstance(close, pd.DataFrame):
    close = close.squeeze()  # turns (n,1) -> (n,)
elif isinstance(close.values[0], (np.ndarray, list)):
    close = close.apply(lambda x: x[0])

# Calculate indicators
data['RSI'] = ta.momentum.RSIIndicator(close).rsi()
macd = ta.trend.MACD(close)
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()
data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], close).adx()
data['Volume_SMA'] = data['Volume'].rolling(window=14).mean()

# Buy/Sell Signals (simple logic)
data['Buy'] = (data['RSI'] < 30) & (data['MACD'] > data['MACD_signal']) & (data['ADX'] > 20)
data['Sell'] = (data['RSI'] > 70) & (data['MACD'] < data['MACD_signal']) & (data['ADX'] > 20)

# Plotting
st.subheader(f"📉 Price & Indicators for {symbol}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Close'], label='Close Price')

# Buy/Sell markers
ax.plot(data.index[data['Buy']], data['Close'][data['Buy']], '^', markersize=10, color='green', label='Buy')
ax.plot(data.index[data['Sell']], data['Close'][data['Sell']], 'v', markersize=10, color='red', label='Sell')

ax.set_title(f"{symbol} Price with Buy/Sell Signals")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid()
st.pyplot(fig)

# Show data
with st.expander("📋 Show raw data"):
    st.dataframe(data.tail(100))

# Optional download
csv = data.to_csv().encode('utf-8')
st.download_button("📥 Download Data", data=csv, file_name=f'{symbol}_data.csv')
