import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from streamlit_autorefresh import st_autorefresh
from src.prediction.predict import predict_today

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st.title("📈 AI Trading Dashboard")

# =========================
# AUTO REFRESH
# =========================
st_autorefresh(interval=300000, key="refresh")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    return (
        joblib.load("artifacts/scaler.pkl"),
        joblib.load("artifacts/models.pkl"),
        joblib.load("artifacts/features.pkl")
    )

@st.cache_data(ttl=3600)
def load_data():
    df = joblib.load("artifacts/final_df.pkl")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

final_df = load_data()
scaler, models, FEATURES = load_models()

# =========================
# PREDICT
# =========================
latest_data = final_df.groupby("Company").tail(1)

@st.cache_data(ttl=300)
def run_prediction(data, _scaler, _models, _features):
    signals, portfolio = predict_today(data, _scaler, _models, None, _features)

    if signals is None or signals.empty:
        signals = pd.DataFrame()

    if portfolio is None or portfolio.empty:
        portfolio = signals.copy()

    return signals, portfolio

signals, portfolio = run_prediction(latest_data, scaler, models, FEATURES)

# =========================
# REQUIRED COLUMNS ONLY
# =========================
required_cols = [
    "Company","Signal","Signal_Date","Action_Date","Close","Volume",
    "Probability","Confidence","Position","Target","Exit_Date",
    "Stop_Loss","Expected_return(%)",
    "Risk","Reward","RR_Ratio","Side",
    "Market_Regime","Regime_Strength"
]

signals = signals[[c for c in required_cols if c in signals.columns]]

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔍 Filters")

signal_filter = st.sidebar.selectbox(
    "Signal Type",
    ["ALL","STRONG BUY","BUY","HOLD","SELL","STRONG SELL"]
)

prob_filter = st.sidebar.slider(
    "Minimum Probability", 0.0, 1.0, 0.0
)

filtered = signals.copy()

if signal_filter != "ALL" and "Signal" in filtered.columns:
    filtered = filtered[filtered["Signal"] == signal_filter]

if "Probability" in filtered.columns:
    filtered = filtered[filtered["Probability"] >= prob_filter]

# =========================
# REFRESH BUTTON
# =========================
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# =========================
# COLOR STYLES (ZAZZY)
# =========================
def color_signal(val):

    styles = {
        "STRONG BUY": "background-color:#1B5E20;color:white;font-weight:600",
        "BUY": "background-color:#66BB6A;color:black;font-weight:500",
        "HOLD": "background-color:#FFE082;color:black;font-weight:500",
        "SELL": "background-color:#EF5350;color:white;font-weight:500",
        "STRONG SELL": "background-color:#B71C1C;color:white;font-weight:600"
    }

    return styles.get(val, "")


def color_side(val):
    if val == "LONG":
        return "color:#2E7D32;font-weight:600"
    elif val == "SHORT":
        return "color:#C62828;font-weight:600"
    return ""


def highlight_company(val):
    return "font-weight:600;color:#3B4953"


# =========================
# FORMAT
# =========================
def format_df(df):
    df = df.copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else x
        )

    for col in ["Signal_Date","Action_Date","Exit_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    return df

# =========================
# KPIs
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Stocks", len(signals))
c2.metric("Active Trades", len(portfolio))
c3.metric("LONG", len(portfolio[portfolio["Side"]=="LONG"]) if "Side" in portfolio else 0)
c4.metric("SHORT", len(portfolio[portfolio["Side"]=="SHORT"]) if "Side" in portfolio else 0)

# =========================
# SIGNAL TABLE
# =========================
st.subheader("📊 Trading Signals")

styled = format_df(filtered).style \
    .applymap(color_signal, subset=["Signal"] if "Signal" in filtered else None) \
    .applymap(color_side, subset=["Side"] if "Side" in filtered else None) \
    .applymap(highlight_company, subset=["Company"] if "Company" in filtered else None)

st.dataframe(styled, use_container_width=True)

# =========================
# PORTFOLIO SPLIT
# =========================
st.subheader("💼 Portfolio")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 LONG POSITIONS")
    st.dataframe(
        format_df(portfolio[portfolio["Side"]=="LONG"])
        if "Side" in portfolio else pd.DataFrame(),
        use_container_width=True
    )

with col2:
    st.markdown("### 🔴 SHORT POSITIONS")
    st.dataframe(
        format_df(portfolio[portfolio["Side"]=="SHORT"])
        if "Side" in portfolio else pd.DataFrame(),
        use_container_width=True
    )

# =========================
# PERFORMANCE
# =========================
@st.cache_data
def compute_performance(df):

    df = df.copy().sort_values(["Company","Date"])

    if "EMA_50" not in df.columns:
        df["EMA_50"] = df.groupby("Company")["Close"].transform(lambda x: x.ewm(span=50).mean())

    df["Return"] = df.groupby("Company")["Close"].pct_change()
    df["Signal"] = np.where(df["Close"] > df["EMA_50"], 1, -1)
    df["Strategy"] = df["Signal"] * df["Return"]

    daily = df.groupby("Date")["Strategy"].mean().fillna(0)

    equity = (1 + daily).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax()

    return equity, daily, drawdown

equity, pnl, drawdown = compute_performance(final_df)

st.subheader("📈 Performance")

fig = go.Figure()

fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity"))
fig.add_trace(go.Scatter(x=pnl.index, y=pnl, name="Returns", yaxis="y2"))

fig.update_layout(
    template="plotly_dark",
    yaxis=dict(title="Equity"),
    yaxis2=dict(title="Returns", overlaying="y", side="right")
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DRAWDOWN
# =========================
st.subheader("📉 Drawdown")

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy"))
fig_dd.update_layout(template="plotly_dark")

st.plotly_chart(fig_dd, use_container_width=True)

# =========================
# STOCK CHART
# =========================
st.subheader("📊 Stock Chart")

company = st.selectbox("Select Company", final_df["Company"].unique())

data = final_df[final_df["Company"] == company].copy()

# EMA selection
ema_options = st.multiselect(
    "Select EMA",
    ["EMA_10","EMA_20","EMA_50","EMA_200"],
    default=["EMA_10","EMA_50"]
)

# Compute EMAs if missing
for ema in ema_options:
    span = int(ema.split("_")[1])
    if ema not in data.columns:
        data[ema] = data["Close"].ewm(span=span).mean()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data["Date"],
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"]
))

for ema in ema_options:
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data[ema],
        name=ema
    ))

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)
