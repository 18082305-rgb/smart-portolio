import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="ARAS - Smart Portfolio", layout="wide")

# ------------------------------
# Animated Top Ticker + Nav Bar
# ------------------------------
st.markdown("""
<style>
/* ===== NAV BAR ===== */
.top-bar {
    background-color: #D6E6F2;
    padding: 8px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #A9CFE7;
    font-family: Arial, sans-serif;
    font-size: 14px;
    border-radius: 10px;
}
.top-bar a {
    text-decoration: none;
    color: #1A4D80;
    font-weight: 600;
    margin-left: 14px;
}
.top-bar a:hover { color: #0D2B4F; }
.top-title {
    font-weight: 900;
    color: #1A4D80;
}

/* ===== TICKER (Moving Ad) ===== */
.ticker-wrap {
    background: #0B2447;
    color: #ffffff;
    overflow: hidden;
    white-space: nowrap;
    border-radius: 10px;
    margin: 10px 0 14px 0;
    border: 1px solid rgba(255,255,255,0.18);
}
.ticker {
    display: inline-block;
    padding-left: 100%;
    animation: tickerMove 18s linear infinite;
}
.ticker span{
    display: inline-block;
    padding: 10px 28px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.3px;
}
@keyframes tickerMove {
    0% { transform: translate3d(0,0,0); }
    100% { transform: translate3d(-100%,0,0); }
}

/* ===== Big CTA Button ===== */
div.stButton > button {
    background: #1A4D80 !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    padding: 10px 18px !important;
    border-radius: 12px !important;
    border: 0 !important;
}
div.stButton > button:hover {
    background: #0D2B4F !important;
}

/* ===== Fixed Back Home Button ===== */
.fixed-back {
    position: fixed;
    bottom: 18px;
    right: 18px;
    z-index: 999999;
}
.fixed-back button {
    background: #111827 !important;
    color: #fff !important;
    font-weight: 900 !important;
    border-radius: 999px !important;
    padding: 10px 16px !important;
    border: 0 !important;
}
.fixed-back button:hover { background: #000 !important; }

/* Make content not hide behind fixed button */
.block-container { padding-bottom: 90px !important; }
</style>

<div class="top-bar">
    <div class="top-title">📊 ARAS – Smart Portfolio</div>
    <div>
        <a href="https://www.msx.om" target="_blank">📰 Muscat Stock Exchange</a>
        <a href="https://www.omanobserver.om/section/business" target="_blank">📈 Oman Market News</a>
    </div>
</div>

<div class="ticker-wrap">
  <div class="ticker">
    <span>🚀 ARAS helps you monitor MSX faster • Clear AI insights • Confidence score • Compare Omantel & Ooredoo • Save time & invest smarter</span>
    <span>📊 Smart tip: Shorter ranges are fine, but for stronger confidence choose a period with enough data • ARAS keeps it simple and investor-friendly</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Session state
# ------------------------------
if "start_analysis" not in st.session_state:
    st.session_state.start_analysis = False

# ------------------------------
# Helper functions
# ------------------------------
def process_stock_file(file):
    df = pd.read_excel(file)

    # keep rows that look like data
    df = df[df.iloc[:, 0].astype(str).str.contains(r"\d", regex=True)]
    df = df.iloc[:, :6].copy()
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("Date")
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()

def predict_price(df, horizon):
    features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI"]
    X = df[features]
    y = df["Close"].shift(-horizon)

    X = X.iloc[:-horizon] if horizon > 0 else X
    y = y.iloc[:-horizon] if horizon > 0 else y

    # Not enough data → quick mode
    if len(X) < 8 or len(y) < 8:
        # fallback: simple "carry forward" prediction
        # (still gives user result without error)
        predicted = float(df.iloc[-1]["Close"])
        return predicted, None, None, None

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    predicted = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])
    return predicted, model, X, y

def confidence_score(model, X_test, y_test):
    mae = mean_absolute_error(y_test, model.predict(X_test))
    error_conf = max(0, 1 - mae / (y_test.mean() if y_test.mean() != 0 else 1))
    tree_preds = np.array([tree.predict(X_test.iloc[-1].values.reshape(1, -1))[0] for tree in model.estimators_])
    stability = 1 / (1 + np.std(tree_preds))
    confidence = (0.6 * error_conf + 0.4 * stability) * 100
    return float(min(95, max(40, confidence)))

def compare_stocks(df1, name1, df2, name2, horizon):
    def analyze(df, name):
        pred, model, X, y = predict_price(df, horizon)
        last = float(df.iloc[-1]["Close"])
        profit_pct = (pred - last) / last * 100 if last != 0 else 0

        if model is None or X is None or y is None or len(X) < 10:
            conf = 45.0
            mode = "Quick Insight"
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            conf = confidence_score(model, X_test, y_test)
            mode = "AI Model"

        trend = "Up ✅" if profit_pct > 0 else "Down ❌"
        return {
            "Name": name,
            "Mode": mode,
            "Last Close": last,
            "Predicted": float(pred),
            "Profit %": float(profit_pct),
            "Trend": trend,
            "Confidence": float(conf)
        }
    return analyze(df1, name1), analyze(df2, name2)

def pick_period_dates(df, preset):
    max_date = df["Date"].max().date()
    if preset == "Today":
        return max_date, max_date
    if preset == "1 Week":
        return (max_date - timedelta(days=7)), max_date
    if preset == "1 Month":
        return (max_date - timedelta(days=30)), max_date
    if preset == "1 Year":
        return (max_date - timedelta(days=365)), max_date
    if preset == "3 Years":
        return (max_date - timedelta(days=365*3)), max_date
    return df["Date"].min().date(), max_date

# ------------------------------
# Files
# ------------------------------
files_dict = {
    "Omantel.xlsx": "Omantel.xlsx",
    "Ooredoo.xlsx": "Ooredoo.xlsx"
}

# ------------------------------
# Welcome Page
# ------------------------------
if not st.session_state.start_analysis:
    st.markdown("<h1 style='text-align:center;color:#1A4D80;font-size:54px;font-weight:900;'>Welcome to ARAS</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;color:#1A4D80;font-size:24px;font-weight:700;'>An Intelligent Investment Recommendation System for the Oman Stock Market</h5>", unsafe_allow_html=True)
    st.markdown("---")

    ads = [
        ("📈 Predict stock moves before the market reacts!", "#D6E6F2"),
        ("🤖 Confidence score for Omantel & Ooredoo in seconds!", "#D6E6F2"),
        ("📊 Compare top stocks with one selection!", "#D6E6F2"),
        ("💡 Save time, reduce noise, and invest smarter today!", "#D6E6F2"),
    ]
    for text, color in ads:
        st.markdown(f"""
        <div style='background-color:{color};padding:22px;border-radius:14px;margin-bottom:12px;
                    color:#0D2B4F;text-align:center;font-size:22px;font-weight:900;border:1px solid #A9CFE7;'>
            {text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("🚀 Start Analysis"):
        st.session_state.start_analysis = True
        st.rerun()

# ------------------------------
# Main Analysis Page
# ------------------------------
if st.session_state.start_analysis:
    st.success("ARAS Loaded! Choose your company and time period below.")

    # Investor chooses company
    stock_choice = st.selectbox("Select Company", list(files_dict.keys()))

    # Load selected stock
    df_full = process_stock_file(files_dict[stock_choice])

    st.markdown("## Choose Time Period (Investor Selection)")
    preset = st.radio(
        "Select analysis period",
        ["Today", "1 Week", "1 Month", "1 Year", "3 Years", "Custom (Calendar)"],
        horizontal=True
    )

    min_date = df_full["Date"].min().date()
    max_date = df_full["Date"].max().date()

    # Default dates based on preset (but user can still change if custom)
    if preset != "Custom (Calendar)":
        s_default, e_default = pick_period_dates(df_full, preset)
        s_default = max(s_default, min_date)
        e_default = min(e_default, max_date)
        start_date = s_default
        end_date = e_default
        st.info(f"Selected range: {start_date} → {end_date}")
    else:
        st.caption("Select Start and End Dates")
        start_date, end_date = st.date_input(
            "Select Start and End Dates",
            value=[max_date - timedelta(days=7), max_date],
            min_value=min_date,
            max_value=max_date
        )

    # Marketing / investor-friendly note
    st.info("Smart Tip: Shorter ranges are great for quick checks. For higher-confidence insights, choose a period that includes enough data (e.g., 1–3 months).")

    # Horizon days
    horizon_days = (end_date - start_date).days
    if horizon_days < 1:
        horizon_days = 1

    # Clip horizon to data length safely (no errors)
    max_horizon = max(1, len(df_full) - 15)
    if horizon_days > max_horizon:
        horizon_days = max_horizon

    # Slice data to selected window (but keep some history to compute indicators)
    df_window = df_full[(df_full["Date"].dt.date >= start_date) & (df_full["Date"].dt.date <= end_date)].copy()

    # If user selected a tiny window, use fallback history to still run model
    if len(df_window) < 15:
        df_window = df_full.tail(90).copy()

    # ------------------------------
    # AUTO REPORT (no button)
    # ------------------------------
    st.markdown("### ✅ ARAS Report (Auto)")

    predicted_price, model, X, y = predict_price(df_window, horizon_days)

    current_price = float(df_full.iloc[-1]["Close"])
    profit_pct = (predicted_price - current_price) / current_price * 100 if current_price != 0 else 0
    future_date = df_full.iloc[-1]["Date"] + pd.Timedelta(days=horizon_days)

    if model is None or X is None:
        confidence = 45.0
        mode_label = "Quick Insight Mode"
    else:
        if len(X) < 10:
            confidence = 50.0
            mode_label = "AI Model Mode (Limited Data)"
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            confidence = confidence_score(model, X_test, y_test)
            mode_label = "AI Model Mode"

    if profit_pct > 1:
        recommendation, rec_color = "Buy 📈", "#00AA00"
    elif profit_pct < -1:
        recommendation, rec_color = "Avoid/Sell 📉", "#FF0000"
    else:
        recommendation, rec_color = "Hold ⚪", "#FF8C00"

    st.subheader(f"Stock Report: {stock_choice}")
    st.write(f"**Mode:** {mode_label}")
    st.write(f"**Selected Period:** {start_date} → {end_date}  (**{horizon_days} days**)")

    st.write(f"**Current Price:** {current_price:.3f} OMR")
    st.write(f"**Predicted Price ({horizon_days} days):** {predicted_price:.3f} OMR")
    st.write(f"**Profit Expectation:** {profit_pct:.2f}%")
    st.write(f"**Confidence Score:** {confidence:.1f}%")
    st.markdown(f"**Recommendation:** <span style='color:{rec_color};font-weight:900;'>{recommendation}</span>", unsafe_allow_html=True)

    # ---- Actual vs Predicted Chart ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_full["Date"], df_full["Close"], label="Actual Price", color="blue")
    ax.scatter(future_date, predicted_price, color="purple", s=90, label="Predicted Price")
    ax.set_title(f"Actual vs Predicted Price: {stock_choice}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (OMR)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ---- Compare Omantel vs Ooredoo ----
    df_omantel = process_stock_file(files_dict["Omantel.xlsx"])
    df_ooredoo = process_stock_file(files_dict["Ooredoo.xlsx"])

    stock1, stock2 = compare_stocks(df_omantel, "Omantel", df_ooredoo, "Ooredoo", horizon_days)
    st.subheader("Stock Comparison: Omantel vs Ooredoo")
    st.write(pd.DataFrame([stock1, stock2]))

    # ---- Bar chart ----
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(["Omantel", "Ooredoo"], [stock1["Profit %"], stock2["Profit %"]],
            color=["green" if stock1["Profit %"] > 0 else "red",
                   "green" if stock2["Profit %"] > 0 else "red"])
    ax2.set_ylabel("Expected Profit/Loss (%)")
    ax2.set_title("Expected Profit/Loss per Stock")
    ax2.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig2)

    # Closing marketing line
    st.success("🚀 Invest smarter with ARAS: Save time, monitor MSX trends, and act with clearer confidence.")

    # ------------------------------
    # Fixed Back to Home Button
    # ------------------------------
    back_ph = st.empty()
    with back_ph.container():
        st.markdown('<div class="fixed-back">', unsafe_allow_html=True)
        if st.button("🏠 Back to Home"):
            st.session_state.start_analysis = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
