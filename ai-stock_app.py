import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt

# --- Streamlit UI ---
st.title("📈 AI Stock Analyzer for Penny Stocks")
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt

# --- Streamlit UI ---
st.set_page_config(layout="centered")
st.title("📈 AI Stock Analyzer for Penny Stocks")

symbol = st.text_input("Enter Stock Symbol (e.g., SNDL, GME):", "SNDL")

# --- Load Data ---
data = yf.download(symbol, period="6mo", interval="1d")

# --- Έλεγχος για δεδομένα ---
if data.empty or data['Close'].isnull().all():
    st.error("⚠️ No price data available for this symbol. Try a different one.")
    st.stop()

data = data.dropna(subset=['Close'])  # Καθαρισμός NaN τιμών

# --- Indicators ---
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
macd = ta.trend.MACD(data['Close'])
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()
data['MACD_hist'] = macd.macd_diff()
adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'])
data['ADX'] = adx.adx()

# --- Target: Will price go UP in 3 days? ---
data['Future_Close'] = data['Close'].shift(-3)
data['Target'] = (data['Future_Close'] > data['Close']).astype(int)

# --- Features & Model ---
features = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ADX', 'Volume']
df = data.dropna()  # αφαιρεί σειρές με NaN

if len(df) < 50:
    st.error("⚠️ Not enough data after indicators. Try different stock or timeframe.")
    st.stop()

X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: {acc*100:.2f}%")

# --- Prediction for latest day ---
latest_data = df[features].iloc[-1:]
pred = model.predict(latest_data)[0]
proba = model.predict_proba(latest_data)[0][1]

if pred == 1:
    st.markdown(f"### 🔼 AI Suggests: **BUY** (Confidence: {proba*100:.1f}%)")
else:
    st.markdown(f"### 🔽 AI Suggests: **HOLD / SELL** (Confidence: {(1-proba)*100:.1f}%)")

# --- Chart ---
st.subheader("Price Chart & Volume")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Close'], label='Close Price', color='blue', alpha=0.6)
ax.set_ylabel("Price")
ax2 = ax.twinx()
ax2.bar(data.index, data['Volume'], alpha=0.2, label="Volume", color='gray')
ax.legend(loc="upper left")
st.pyplot(fig)0:.2f}%")

# --- Predict latest day ---
latest_data = df[features].iloc[-1:]
pred = model.predict(latest_data)[0]
proba = model.predict_proba(latest_data)[0][1]

if pred == 1:
    st.markdown(f"### 🔼 AI Suggests: **BUY** (Confidence: {proba*100:.1f}%)")
else:
    st.markdown(f"### 🔽 AI Suggests: **HOLD / SELL** (Confidence: {(1-proba)*100:.1f}%)")

# --- Chart ---
st.subheader("Price Chart & Volume")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Close'], label='Close Price')
ax.set_ylabel("Price")
ax2 = ax.twinx()
ax2.bar(data.index, data['Volume'], alpha=0.2, label="Volume", color='gray')
ax.legend(loc="upper left")
st.pyplot(fig)
