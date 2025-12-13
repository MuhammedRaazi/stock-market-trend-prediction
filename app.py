# strong_move_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# Streamlit UI Setup
# --------------------------------------------------
st.set_page_config(page_title="Stock Move Direction Predictor",
                   layout="wide", page_icon="📈")

st.markdown("""
<style>
.stApp { background: #0b1220; color: #e6eef8; }
.block-container { padding-top: 1rem; }
.prediction-card { background:#0f1a2b; border-radius:12px; padding:18px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model + Feature List + Ticker Categories
# --------------------------------------------------
MODEL_PATH = "StrongMove_XGB_Model.pkl"
FEATURE_LIST_PATH = "features.pkl"
TICKER_CATS_PATH = "ticker_categories.pkl"

model = joblib.load(MODEL_PATH)
FEATURES = list(joblib.load(FEATURE_LIST_PATH))
ticker_categories = joblib.load(TICKER_CATS_PATH)

# --------------------------------------------------
# Helper
# --------------------------------------------------
def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def remove_incomplete_daily(df):
    """Remove only today's incomplete candle."""
    if df.empty:
        return df

    latest = df.index[-1].normalize()
    today = pd.Timestamp.today().normalize()

    if latest == today:
        df = df.iloc[:-1]   # remove incomplete candle

    return df


@st.cache_data(ttl=3600)
def load_nifty(period_days=365):
    n = yf.download("^NSEI", period=f"{period_days}d",
                    interval="1d", auto_adjust=False, progress=False)
    if n is None or n.empty:
        return pd.DataFrame()

    n = _flatten(n)
    n = remove_incomplete_daily(n)

    n["NIFTY_Return_1D"] = n["Close"].pct_change()
    n["NIFTY_Vol_10D"] = n["NIFTY_Return_1D"].rolling(10).std()

    return n[["NIFTY_Return_1D", "NIFTY_Vol_10D"]]


@st.cache_data(ttl=1200)
def build_features_df(stock, period_days=180):
    df = yf.download(stock, period=f"{period_days}d",
                     interval="1d", auto_adjust=False, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten(df)
    df = remove_incomplete_daily(df)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Strong Move Features
    df["Return_1D"] = df["Close"].pct_change()
    df["Return_2D"] = df["Close"].pct_change(2)
    df["Return_3D"] = df["Close"].pct_change(3)

    df["Momentum_10D"] = df["Close"] - df["Close"].shift(10)

    df["RSI_14"] = ta.rsi(df["Close"], 14)
    df["RSI_14_lag1"] = df["RSI_14"].shift(1)

    df["Volatility_10D"] = df["Return_1D"].rolling(10).std()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_10"] = ta.ema(df["Close"], length=10)

    df["Candle_Body%"] = (df["Close"] - df["Open"]) / df["Open"]
    df["High_Low_Range%"] = (df["High"] - df["Low"]) / df["Close"]
    df["Volume_Change%"] = df["Volume"].pct_change()

    # NIFTY Join
    nifty = load_nifty(period_days)
    if not nifty.empty:
        df = df.join(nifty, how="left")

    df = df.dropna()
    return df


def build_X_live(stock):
    df = build_features_df(stock)

    if df.empty:
        raise Exception("No usable feature rows.")

    # Encode ticker
    code = pd.Categorical([stock], categories=list(ticker_categories)).codes[0]
    df["TickerEncoded"] = code

    last = df.iloc[-1:].copy()

    # Ensure all model features exist
    for f in FEATURES:
        if f not in last.columns:
            last[f] = 0.0

    X = last[FEATURES].astype(float)
    return X, df


def predict_strong_move(stock):
    X, df_all = build_X_live(stock)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return pred, proba, X, df_all


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙ Controls")
    stock = st.selectbox("Select Stock", list(ticker_categories))

    if st.button("Predict"):
        st.session_state["run_predict"] = True


# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("NIFTY-50 Strong Move Direction Prediction📈")

left, right = st.columns([1.2, 2])

# Prediction Box
with left:
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown("### Model Prediction")

    if st.session_state.get("run_predict", False):

        pred, proba, X_live, df_all = predict_strong_move(stock)

        if pred == 1:
            text = "STRONG UP SIGNAL"
            color = "#2ecc71"
            conf = proba[1] * 100
        else:
            text = "STRONG DOWN SIGNAL"
            color = "#ff4b4b"
            conf = proba[0] * 100

        st.markdown(f"## {stock} → **{text}**")

        bar_html = f"""
        <div style="margin-top:10px; width:100%; background:#1e2b3a;
                    border-radius:10px; padding:4px;">
            <div style="width:{conf}%; background:{color};
                        height:18px; border-radius:8px;">
            </div>
        </div>
        <p style="margin-top:6px; font-size:16px;">
        <b>Confidence:</b> {conf:.2f}%
        </p>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        st.session_state["latest"] = (X_live, df_all)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# Charts + Tables
# --------------------------------------------------
with right:
    st.markdown("### 📊 Chart & Indicators")

    if "latest" in st.session_state:

        X_live, df_all = st.session_state["latest"]
        last30 = df_all.tail(30)

        fig, ax = plt.subplots(2, 1, figsize=(10, 5),
                               gridspec_kw={'height_ratios': [3, 1]},
                               sharex=True)

        ax[0].plot(last30.index, last30["Close"], label="Close", linewidth=1.4)
        ax[0].plot(last30.index, last30["SMA_5"], label="SMA_5")
        ax[0].plot(last30.index, last30["SMA_10"], label="SMA_10")
        ax[0].legend(loc="upper left")

        ax[1].plot(last30.index, last30["RSI_14"], label="RSI_14")
        ax[1].axhline(70, color="red", linewidth=0.6)
        ax[1].axhline(30, color="green", linewidth=0.6)

        st.pyplot(fig)

        st.markdown("###  Latest Feature Snapshot")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Raw Indicators")
            st.dataframe(df_all.iloc[-1:].T)

        with col2:
            st.markdown("#### Model Input")
            st.dataframe(X_live.T)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Strong Move Predictor © 2025")