import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="ARAS | Smart Portfolio",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# -----------------------------
# Files Preloaded
# -----------------------------
FILES = {
    "Omantel": "data/omantel.xlsx",
    "Ooredoo": "data/ooredoo.xlsx"
}

# -----------------------------
# Navbar
# -----------------------------
def top_bar():
    col1, col2, col3 = st.columns([3,3,1])
    with col1:
        st.markdown("### 📈 Muscat Market Today")
        st.caption("Market Status: Stable | Update: Live")
    with col2:
        st.markdown("### 💰 OMAN Index")
        st.caption("Last Price: 4,520 | Change: +0.6%")
    with col3:
        if st.button("🚪 Exit App"):
            st.stop()

# -----------------------------
# Load Stock Data
# -----------------------------
def compute_RSI(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_stock(stock_name):
    df = pd.read_excel(FILES[stock_name])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["RSI"] = compute_RSI(df["Close"],14)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------------
# Dashboard Page
# -----------------------------
def dashboard_page():
    top_bar()
    st.markdown("---")
    st.markdown("## 📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Market Trend", "Bullish", "+1.2%")
    col2.metric("Top Stock", "Omantel", "+2.4%")
    col3.metric("Risk Level", "Medium")

    st.markdown("### Quick Actions")
    col4, col5 = st.columns(2)
    with col4:
        if st.button("📈 Stock Analysis"):
            st.session_state.page = "analysis"
    with col5:
        if st.button("📊 Stock Comparison"):
            st.session_state.page = "comparison"

# -----------------------------
# Stock Analysis Page
# -----------------------------
def analysis_page():
    top_bar()
    st.markdown("---")
    st.markdown("## 📈 Stock Analysis")

    stock = st.selectbox("Select Stock", list(FILES.keys()))
    df_full = load_stock(stock)
    min_date = df_full["Date"].min()
    max_date = df_full["Date"].max()

    # اختيار الفترة
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
    with col2:
        end_date = st.date_input("To", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

    if start_date > end_date:
        st.error("Start date must be before End date")
        return

    df = df_full[(df_full["Date"].dt.date >= start_date) & (df_full["Date"].dt.date <= end_date)].copy()
    if len(df)<10:
        st.warning("Not enough data for this period")
        return

    # Features & Model
    X = df[["MA5","MA10","RSI"]]
    y = df["Close"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X,y)

    prediction = model.predict(X.tail(1))[0]
    current = df.iloc[-1]["Close"]
    expected_return = (prediction-current)/current*100

    # Indicators
    rsi_val = df.iloc[-1]["RSI"]
    rsi_status = ("Overbought" if rsi_val>70 else "Oversold" if rsi_val<30 else "Normal")
    trend = "Uptrend" if df["Close"].iloc[-1] > df["Close"].iloc[-5] else "Downtrend"
    risk = "Stable" if df["Close"].pct_change().std()<0.02 else "High Volatility"
    recommendation = "BUY" if expected_return>3 else "Avoid/Sell" if expected_return<-3 else "HOLD"

    # Display Info
    st.markdown(f"""
    <div style='background:#F9FAFB;padding:20px;border-radius:12px;border:1px solid #E5E7EB'>
    <b>Current Price:</b> {current:.3f} OMR<br>
    <b>Predicted Price:</b> {prediction:.3f} OMR<br>
    <b>Expected Return:</b> {expected_return:.2f}%<br>
    <b>RSI:</b> {rsi_status}<br>
    <b>Trend:</b> {trend}<br>
    <b>Risk:</b> {risk}<br>
    <b>Recommendation:</b> {recommendation}
    </div>
    """, unsafe_allow_html=True)

    # Chart
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Date"], df["Close"], label="Close Price", color="blue")
    ax.plot(df["Date"], df["MA5"], label="MA5", linestyle="--", color="green")
    ax.plot(df["Date"], df["MA10"], label="MA10", linestyle="--", color="orange")
    ax.scatter(df["Date"].iloc[-1], prediction, color="red", s=150, label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (OMR)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"

# -----------------------------
# Stock Comparison Page
# -----------------------------
def comparison_page():
    top_bar()
    st.markdown("---")
    st.markdown("## 📊 Stock Comparison")

    report_data=[]
    for stock in FILES.keys():
        df = load_stock(stock)
        X = df[["MA5","MA10","RSI"]]
        y = df["Close"]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X,y)
        prediction = model.predict(X.tail(1))[0]
        current = df.iloc[-1]["Close"]
        expected_return = (prediction-current)/current*100
        rsi_val = df.iloc[-1]["RSI"]
        rsi_status = ("Overbought" if rsi_val>70 else "Oversold" if rsi_val<30 else "Normal")
        trend = "Uptrend" if df["Close"].iloc[-1]>df["Close"].iloc[-5] else "Downtrend"
        risk = "Stable" if df["Close"].pct_change().std()<0.02 else "High Volatility"
        recommendation = "BUY" if expected_return>3 else "Avoid/Sell" if expected_return<-3 else "HOLD"

        report_data.append({
            "Stock": stock,
            "Current Price": round(current,3),
            "RSI": rsi_status,
            "Trend": trend,
            "Period Performance %": round(expected_return,2),
            "Risk": risk,
            "Recommendation": recommendation
        })

    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True)

    best_stock = df_report.sort_values("Period Performance %", ascending=False).iloc[0]
    st.markdown(f"### ⭐ Best Stock Recommendation: **{best_stock['Stock']}** ({best_stock['Recommendation']})")

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"

# -----------------------------
# Routing
# -----------------------------
if st.session_state.page=="dashboard":
    dashboard_page()
elif st.session_state.page=="analysis":
    analysis_page()
elif st.session_state.page=="comparison":
    comparison_page()
