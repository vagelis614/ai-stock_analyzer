import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

symbol = st.text_input("📊 Enter Stock Ticker", "AAPL").upper()

if not symbol:
    st.warning("Please enter a stock symbol.")
    st.stop()

# Download stock data
data = yf.download(symbol, period="6mo", interval="1d")

# 🔧 Ελέγχει αν υπάρχει έγκυρο ιστορικό για τον ticker
if data is None or data.empty or data['Close'].dropna().empty:
    st.error("No valid data found for the given symbol.")
    st.stop()

# --- Indicators ---
data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'].squeeze()).rsi()
macd = ta.trend.MACD(close=data['Close'].squeeze())
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()
data['ADX'] = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'].squeeze()).adx()
data['Volume_avg'] = data['Volume'].rolling(window=14).mean()

# --- Signals ---
data['Buy'] = (data['RSI'] < 30) & (data['MACD'] > data['MACD_signal']) & (data['ADX'] > 20)
data['Sell'] = (data['RSI'] > 70) & (data['MACD'] < data['MACD_signal']) & (data['ADX'] > 20)

# --- Chart ---
st.subheader(f"📉 Price Chart for {symbol}")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)

# Add Buy signals
ax.plot(data.index[data['Buy']], data['Close'][data['Buy']], '^', markersize=10, color='green', label='Buy Signal')

# Add Sell signals
ax.plot(data.index[data['Sell']], data['Close'][data['Sell']], 'v', markersize=10, color='red', label='Sell Signal')

ax.set_title(f"{symbol} Price & Signals")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Show indicators ---
with st.expander("📊 Technical Indicators"):
    st.dataframe(data[['RSI', 'MACD', 'MACD_signal', 'ADX', 'Volume', 'Volume_avg']].tail(20))

# --- Show recommendations ---
last_row = data.iloc[-1]
st.markdown("## 💡 Signal Summary")
if last_row['Buy']:
    st.success("✅ Recommendation: **BUY** signal detected.")
elif last_row['Sell']:
    st.error("🚫 Recommendation: **SELL** signal detected.")
else:
    st.info("⏸️ Recommendation: No strong signal detected at this time.")
