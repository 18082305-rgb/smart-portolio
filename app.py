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
# Files Preloaded (مطابقة لـ GitHub)
# -----------------------------
FILES = {
    "Omantel": "Omantel.xlsx",
    "Ooredoo": "Ooredoo.xlsx"
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
# RSI Calculation
# -----------------------------
def compute_RSI(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------------
# Load Stock Data
# -----------------------------
def load_stock(stock_name):
    df = pd.read_excel(FILES[stock_name])
    df["التاريخ"] = pd.to_datetime(df["التاريخ"])
    df = df.sort_values("التاريخ")

    df["MA5"] = df["الإغلاق"].rolling(5).mean()
    df["MA10"] = df["الإغلاق"].rolling(10).mean()
    df["RSI"] = compute_RSI(df["الإغلاق"])

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------------
# Routing
# -----------------------------
if st.session_state.page == "dashboard":
    dashboard_page()
elif st.session_state.page == "analysis":
    analysis_page()
elif st.session_state.page == "comparison":
    comparison_page()
