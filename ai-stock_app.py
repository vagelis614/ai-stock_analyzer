import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# --- User input ---
symbol = st.text_input("Enter stock ticker (e.g. AAPL, TSLA, AMD):", "AAPL").upper()

# --- Download data ---
try:
    data = yf.download(symbol, period="6mo", interval="1d")
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

# --- Check data validity ---
if data.empty or 'Close' not in data.columns:
    st.error("No valid data found for the given symbol.")
    st.stop()

# --- Handle dimensionality issues ---
def ensure_series(series):
    if isinstance(series, pd.DataFrame):
        return series.squeeze()
    return series

# Make sure all are Series
close = ensure_series(data['Close'])
high = ensure_series(data['High'])
low = ensure_series(data['Low'])
volume = ensure_series(data['Volume'])

# --- Technical Indicators ---
data['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
macd = ta.trend.MACD(close=close)
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()
adx = ta.trend.ADXIndicator(high=high, low=low, close=close)
data['ADX'] = adx.adx()

# --- Plot Price & Indicators ---
st.subheader(f"Price Chart for {symbol}")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, close, label="Close Price")
ax.set_title(f"{symbol} Closing Price")
ax.legend()
st.pyplot(fig)

# --- RSI Plot ---
st.subheader("RSI (Relative Strength Index)")
fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
ax_rsi.plot(data.index, data['RSI'], label="RSI", color="orange")
ax_rsi.axhline(70, color="red", linestyle="--")
ax_rsi.axhline(30, color="green", linestyle="--")
ax_rsi.set_title("RSI")
ax_rsi.legend()
st.pyplot(fig_rsi)

# --- MACD Plot ---
st.subheader("MACD (Moving Average Convergence Divergence)")
fig_macd, ax_macd = plt.subplots(figsize=(12, 3))
ax_macd.plot(data.index, data['MACD'], label="MACD", color="blue")
ax_macd.plot(data.index, data['MACD_signal'], label="Signal", color="red")
ax_macd.set_title("MACD")
ax_macd.legend()
st.pyplot(fig_macd)

# --- ADX Plot ---
st.subheader("ADX (Average Directional Index)")
fig_adx, ax_adx = plt.subplots(figsize=(12, 3))
ax_adx.plot(data.index, data['ADX'], label="ADX", color="purple")
ax_adx.axhline(20, linestyle="--", color="gray")
ax_adx.set_title("ADX")
ax_adx.legend()
st.pyplot(fig_adx)

# --- Volume Plot ---
st.subheader("Volume")
fig_vol, ax_vol = plt.subplots(figsize=(12, 3))
ax_vol.bar(data.index, volume, label="Volume", color="gray")
ax_vol.set_title("Volume")
ax_vol.legend()
st.pyplot(fig_vol)

# --- Basic Buy/Sell Signal (for demo) ---
st.subheader("🔍 Basic Signal")
latest_rsi = data['RSI'].iloc[-1]
latest_macd = data['MACD'].iloc[-1]
latest_macd_signal = data['MACD_signal'].iloc[-1]

if latest_rsi < 30 and latest_macd > latest_macd_signal:
    st.success("📈 Potential BUY signal detected (RSI < 30 and MACD crossover)")
elif latest_rsi > 70 and latest_macd < latest_macd_signal:
    st.error("📉 Potential SELL signal detected (RSI > 70 and MACD crossover)")
else:
    st.info("⚖️ No strong buy/sell signals right now.")

st.caption("Powered by yfinance, ta, and matplotlib.")
