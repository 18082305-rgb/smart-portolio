import streamlit as st
import pandas as pd
import numpy as np
import bcrypt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="ARAS | Smart Portfolio",
    layout="wide"
)

# ----------------------------------
# Users Database (Mock – later DB)
# ----------------------------------
USERS = {
    "admin": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()),
    "user1": bcrypt.hashpw("aras2025".encode(), bcrypt.gensalt())
}

# ----------------------------------
# Session State
# ----------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# ----------------------------------
# Helper: Login Check
# ----------------------------------
def check_login(username, password):
    if username in USERS:
        return bcrypt.checkpw(password.encode(), USERS[username])
    return False

# ----------------------------------
# Top Navbar
# ----------------------------------
def navbar():
    st.markdown("""
    <style>
    .nav {
        background:#111827;
        padding:14px 30px;
        border-radius:12px;
        display:flex;
        justify-content:space-between;
        margin-bottom:25px;
    }
    .nav-title {color:white;font-size:22px;font-weight:600;}
    .nav-links span {color:#D1D5DB;margin-left:25px;}
    </style>

    <div class="nav">
        <div class="nav-title">ARAS</div>
        <div class="nav-links">
            <span>Market News</span>
            <span>Today Price</span>
            <span>Profile</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================
# LOGIN PAGE
# ==================================
if st.session_state.page == "login":

    st.markdown("<h1 style='text-align:center;'>Login to ARAS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Secure access to smart portfolio analysis</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if check_login(username, password):
                st.session_state.authenticated = True
                st.session_state.page = "home"
                st.success("Login successful")
            else:
                st.error("Invalid username or password")

# ==================================
# HOME PAGE
# ==================================
if st.session_state.authenticated and st.session_state.page == "home":

    navbar()

    col_logout, col_title = st.columns([1,6])
    with col_logout:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.page = "login"

    with col_title:
        st.markdown("<h2>Welcome to ARAS Dashboard</h2>", unsafe_allow_html=True)

    st.markdown("---")

    features = [
        "Predict stock prices before the market moves",
        "AI-powered confidence scores",
        "Compare Omantel & Ooredoo",
        "Professional decision support"
    ]

    for f in features:
        st.markdown(
            f"<div style='background:#111827;color:white;padding:25px;border-radius:15px;font-size:22px;margin-bottom:15px;text-align:center;'>"
            f"{f}</div>", unsafe_allow_html=True)

    if st.button("Go to Analysis"):
        st.session_state.page = "analysis"

# ==================================
# ANALYSIS PAGE
# ==================================
if st.session_state.authenticated and st.session_state.page == "analysis":

    navbar()

    col_back, col_logout = st.columns([1,1])
    with col_back:
        if st.button("← Back Home"):
            st.session_state.page = "home"

    with col_logout:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.page = "login"

    st.markdown("<h2>Stock Analysis</h2>", unsafe_allow_html=True)
    st.markdown("---")

    files = {
        "Omantel": "Omantel.xlsx",
        "Ooredoo": "Ooredoo.xlsx"
    }

    stock = st.selectbox("Select Stock", list(files.keys()))
    horizon = st.selectbox("Prediction Horizon", [1,5,22,252])

    def load_data(file):
        df = pd.read_excel(file)
        df = df[df.iloc[:,0].astype(str).str.contains(r"\d")]
        df.columns = ["Date","Open","High","Low","Close","Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        for c in ["Open","High","Low","Close","Volume"]:
            df[c] = pd.to_numeric(df[c])
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df.dropna()

    df = load_data(files[stock])
    X = df[["Open","High","Low","Volume","MA5","MA10","RSI"]]
    y = df["Close"].shift(-horizon)

    X, y = X[:-horizon], y[:-horizon]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    predicted = model.predict(X.iloc[-1].values.reshape(1,-1))[0]
    current = df.iloc[-1]["Close"]
    profit = (predicted-current)/current*100

    st.markdown(f"""
    <div style='background:#F9FAFB;padding:25px;border-radius:15px;border:1px solid #E5E7EB;'>
    <b>Current Price:</b> {current:.3f} OMR<br>
    <b>Predicted Price:</b> {predicted:.3f} OMR<br>
    <b>Expected Return:</b> {profit:.2f}%
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(df["Date"], df["Close"])
    ax.scatter(df.iloc[-1]["Date"], predicted, s=120)
    ax.grid(True)
    st.pyplot(fig)
