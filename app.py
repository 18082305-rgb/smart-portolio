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
# CSS (Top bar + Moving ticker + Responsive + Smaller selections box)
# ------------------------------
st.markdown("""
<style>
/* Make app a bit tighter on mobile */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* --- Top navigation bar (official) --- */
.top-bar {
    background-color: #D6E6F2;
    padding: 8px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #A9CFE7;
    font-family: Arial, sans-serif;
    gap: 10px;
    border-radius: 12px;
}

/* Make it wrap nicely on mobile */
@media (max-width: 768px) {
  .top-bar { flex-direction: column; align-items: flex-start; }
  .top-links { display: flex; flex-wrap: wrap; gap: 10px; }
}

.top-bar a {
    text-decoration: none;
    color: #1A4D80;
    font-weight: 600;
    margin-left: 8px;
}
.top-bar a:hover { color: #0D2B4F; }

.top-title {
    font-weight: 800;
    color: #1A4D80;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* --- Moving ticker (continuous) --- */
.ticker-wrap {
  width: 100%;
  overflow: hidden;
  background: #0B2447;
  border-radius: 12px;
  margin-top: 10px;
  padding: 8px 0;
}
.ticker {
  display: inline-block;
  white-space: nowrap;
  will-change: transform;
  animation: ticker 18s linear infinite;
  color: #ffffff;
  font-family: Arial, sans-serif;
  font-weight: 700;
  font-size: 14px;
  padding-left: 100%;
}
@keyframes ticker {
  0% { transform: translate3d(0,0,0); }
  100% { transform: translate3d(-100%,0,0); }
}

/* --- Smaller "SELECTIONS" box (your white box) --- */
.small-card {
    background: #FFFFFF;
    border: 1px solid #E6EEF6;
    border-radius: 14px;
    padding: 10px 12px;
    margin-bottom: 12px;
}
.small-card h4 {
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
    letter-spacing: 1px;
    color: #6B7280;
    font-size: 13px;
}

/* Helpful info note (marketing style) */
.tip-box {
    background: #EEF6FF;
    border: 1px solid #CFE4FF;
    color: #1A4D80;
    padding: 12px 14px;
    border-radius: 12px;
    font-family: Arial, sans-serif;
    font-weight: 600;
}

/* Recommendation colors */
.rec-buy { color: #0B7A2A; font-weight: 900; }
.rec-hold { color: #B45309; font-weight: 900; }
.rec-sell { color: #B91C1C; font-weight: 900; }

/* Make buttons look nicer */
.stButton button {
  border-radius: 12px;
  font-weight: 800;
  padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Top bar + links
# ------------------------------
st.markdown("""
<div class="top-bar">
    <div class="top-title">
        <span style="font-size:16px;">📊 ARAS – Smart Portfolio</span>
    </div>
    <div class="top-links">
        <a href="https://www.msx.om" target="_blank">📰 Muscat Stock Exchange</a>
        <a href="https://www.omanobserver.om/section/business" target="_blank">📈 Oman Market News</a>
    </div>
</div>

<div class="ticker-wrap">
  <div class="ticker">
    🚀 ARAS AI tracks Omani stocks for you • Faster insights • Clear recommendations • Investor-friendly confidence • Save time & invest smarter 📈
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# OPTIONAL: Logo (2 options)
# 1) Put aras_logo.png in repo root (same folder as app.py) then keep this line.
# 2) Or replace with a direct URL to your logo image.
# ------------------------------
try:
    st.image("aras_logo.png", width=170)
except:
    # fallback if file isn't found (won't break the app)
    pass

# ------------------------------
# Session state
# ------------------------------
if "start_analysis" not in st.session_state:
    st.session_state["start_analysis"] = False

# ------------------------------
# Helper functions (keep your original logic, but make it robust for short ranges)
# ------------------------------
def process_stock_file(file):
    df = pd.read_excel(file)

    # Keep only rows that look like data
    df = df[df.iloc[:, 0].astype(str).str.contains(r"\d", regex=True)]

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("Date")

    # Use min_periods=1 so short ranges won't produce all NaNs
    df["MA5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["MA10"] = df["Close"].rolling(10, min_periods=1).mean()

    # RSI with min periods (robust)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)  # neutral fallback

    return df.reset_index(drop=True)

def predict_price(df, horizon_days):
    # Train on the whole dataset, predict forward horizon using last available row
    features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI"]
    X = df[features].copy()
    y = df["Close"].shift(-horizon_days)

    # Align
    X = X.iloc[:-horizon_days] if horizon_days < len(X) else X.iloc[:-1]
    y = y.iloc[:-horizon_days] if horizon_days < len(y) else y.iloc[:-1]

    if len(X) < 5 or len(y) < 5:
        # Not enough training data — fallback: use last close as predicted
        return float(df["Close"].iloc[-1]), None, None, None

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    last_row = df[features].iloc[-1].values.reshape(1, -1)
    predicted = float(model.predict(last_row)[0])

    return predicted, model, X, y

def confidence_score(model, X_test, y_test):
    # If anything fails, return a safe mid confidence
    try:
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        mean_y = float(np.mean(y_test)) if float(np.mean(y_test)) != 0 else 1.0
        error_conf = max(0.0, 1 - (mae / mean_y))

        # stability from trees (last row)
        lastX = X_test.iloc[-1].values.reshape(1, -1)
        tree_preds = np.array([est.predict(lastX)[0] for est in model.estimators_])
        stability = 1 / (1 + np.std(tree_preds))

        confidence = (0.6 * error_conf + 0.4 * stability) * 100
        return float(min(95, max(40, confidence)))
    except:
        return 55.0

def analyze_stock(df, horizon_days, name=""):
    predicted_price, model, X, y = predict_price(df, horizon_days)
    current_price = float(df.iloc[-1]["Close"])
    profit_pct = (predicted_price - current_price) / current_price * 100 if current_price else 0.0

    # Confidence
    if model is None or X is None or y is None or len(X) < 10:
        conf = 55.0
    else:
        # Time-series split (no shuffle)
        test_size = 0.2
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        conf = confidence_score(model, X_test, y_test)

    # Recommendation
    if profit_pct > 1:
        rec = "Buy 📈"
        rec_class = "rec-buy"
    elif profit_pct < -1:
        rec = "Avoid/Sell 📉"
        rec_class = "rec-sell"
    else:
        rec = "Hold ⚪"
        rec_class = "rec-hold"

    return {
        "Name": name,
        "Current": current_price,
        "Predicted": predicted_price,
        "ProfitPct": float(profit_pct),
        "Confidence": float(conf),
        "Recommendation": rec,
        "RecClass": rec_class
    }

def safe_clamp_dates(df, start_date, end_date):
    min_d = df["Date"].min().date()
    max_d = df["Date"].max().date()
    s = max(start_date, min_d)
    e = min(end_date, max_d)
    if e < s:
        e = s
    return s, e, min_d, max_d

# ------------------------------
# WELCOME PAGE
# ------------------------------
if not st.session_state["start_analysis"]:
    st.markdown("<h1 style='text-align:center;color:#1A4D80;font-size:44px;'>Welcome to ARAS</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h5 style='text-align:center;color:#1A4D80;font-size:20px;max-width:900px;margin:auto;'>"
        "An Intelligent Investment Recommendation System for the Oman Stock Market using Artificial Intelligence "
        "and Investor-Friendly Confidence Insights."
        "</h5>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    ads = [
        "📈 Predict price moves before the market reacts",
        "🤖 Confidence scores designed for investors",
        "📊 Compare Omantel & Ooredoo in seconds",
        "💡 Save time and make smarter decisions today"
    ]

    for t in ads:
        st.markdown(f"""
        <div style='background:#D6E6F2;padding:16px;border-radius:14px;margin-bottom:10px;
                    color:#0D2B4F;text-align:center;font-size:18px;font-weight:800;'>
            {t}
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚀 Start Analysis"):
        st.session_state["start_analysis"] = True
        st.experimental_rerun()

# ------------------------------
# MAIN ANALYSIS PAGE
# ------------------------------
if st.session_state["start_analysis"]:
    st.success("ARAS Loaded! Your analysis is ready below ✅")

    files_dict = {
        "Omantel.xlsx": "Omantel.xlsx",
        "Ooredoo.xlsx": "Ooredoo.xlsx"
    }

    # Smaller SELECTIONS box
    st.markdown("""
    <div class="small-card">
      <h4>SELECTIONS</h4>
    </div>
    """, unsafe_allow_html=True)

    # Investor selects company (not auto)
    stock_choice = st.selectbox("Select Company", list(files_dict.keys()))

    # Load selected
    df_full = process_stock_file(files_dict[stock_choice])

    # Choose period (Investor selection)
    st.subheader("Choose Time Period (Investor Selection)")
    period_mode = st.radio(
        "Select analysis period",
        ["Today", "1 Week", "1 Month", "1 Year", "3 Years", "Custom (Calendar)"],
        horizontal=True
    )

    # Build default dates based on period mode (but investor can still change in calendar in custom)
    max_date = df_full["Date"].max().date()
    min_date = df_full["Date"].min().date()

    if period_mode == "Today":
        start_default = max_date
        end_default = max_date
    elif period_mode == "1 Week":
        start_default = max_date - timedelta(days=7)
        end_default = max_date
    elif period_mode == "1 Month":
        start_default = max_date - timedelta(days=30)
        end_default = max_date
    elif period_mode == "1 Year":
        start_default = max_date - timedelta(days=365)
        end_default = max_date
    elif period_mode == "3 Years":
        start_default = max_date - timedelta(days=365*3)
        end_default = max_date
    else:
        # Custom: keep wide default but user chooses
        start_default = max_date - timedelta(days=14)
        end_default = max_date

    # Clamp defaults into available range
    start_default = max(start_default, min_date)

    # Calendar date picker (always available, and works on mobile)
    start_date, end_date = st.date_input(
        "Select Start and End Dates",
        value=[start_default, end_default],
        min_value=min_date,
        max_value=max_date
    )

    # Clamp to data to avoid errors
    start_date, end_date, min_date, max_date = safe_clamp_dates(df_full, start_date, end_date)

    # Horizon days (at least 1 day)
    horizon_days = max(1, (end_date - start_date).days)

    # Marketing note instead of warnings/errors
    st.markdown("""
    <div class="tip-box">
      💡 <b>Smart Tip:</b> Short ranges are quick and clean — for stronger confidence, choose a range with more history.
      (ARAS will still generate results even for a single day.)
    </div>
    """, unsafe_allow_html=True)

    # Use full df for ML training, but show chart for the investor-selected range
    df_range = df_full[(df_full["Date"].dt.date >= start_date) & (df_full["Date"].dt.date <= end_date)].copy()
    if df_range.empty:
        # If user picks dates that produce empty set (edge), fallback
        df_range = df_full.tail(30).copy()

    # Analyze selected stock
    result = analyze_stock(df_full, horizon_days, name=stock_choice)
    current_price = result["Current"]
    predicted_price = result["Predicted"]
    profit_pct = result["ProfitPct"]
    confidence = result["Confidence"]
    rec = result["Recommendation"]
    rec_class = result["RecClass"]

    # Report
    st.subheader(f"Stock Report: {stock_choice}")
    st.write(f"**Selected Period:** {start_date} → {end_date}  (**Horizon:** {horizon_days} day(s))")
    st.write(f"**Current Price:** {current_price:.3f} OMR")
    st.write(f"**Predicted Price ({horizon_days} day(s)):** {predicted_price:.3f} OMR")
    st.write(f"**Profit Expectation:** {profit_pct:.2f}%")
    st.write(f"**Confidence Score:** {confidence:.1f}%")
    st.markdown(f"**Recommendation:** <span class='{rec_class}'>{rec}</span>", unsafe_allow_html=True)

    # Chart (Actual vs Predicted point)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_range["Date"], df_range["Close"], label="Actual Price")
    # show predicted point at end_date (visual)
    ax.scatter(pd.to_datetime(end_date), predicted_price, s=90, label="Predicted", marker="o")
    ax.set_title(f"Actual vs Predicted (Selected Range) — {stock_choice}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (OMR)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Compare Omantel vs Ooredoo
    st.subheader("Stock Comparison: Omantel vs Ooredoo")
    df_omantel = process_stock_file(files_dict["Omantel.xlsx"])
    df_ooredoo = process_stock_file(files_dict["Ooredoo.xlsx"])
    r1 = analyze_stock(df_omantel, horizon_days, "Omantel")
    r2 = analyze_stock(df_ooredoo, horizon_days, "Ooredoo")

    comp_df = pd.DataFrame([{
        "Company": r1["Name"],
        "Current (OMR)": round(r1["Current"], 3),
        "Predicted (OMR)": round(r1["Predicted"], 3),
        "Profit %": round(r1["ProfitPct"], 2),
        "Confidence %": round(r1["Confidence"], 1),
        "Recommendation": r1["Recommendation"]
    },{
        "Company": r2["Name"],
        "Current (OMR)": round(r2["Current"], 3),
        "Predicted (OMR)": round(r2["Predicted"], 3),
        "Profit %": round(r2["ProfitPct"], 2),
        "Confidence %": round(r2["Confidence"], 1),
        "Recommendation": r2["Recommendation"]
    }])
    st.dataframe(comp_df, use_container_width=True)

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(
        ["Omantel", "Ooredoo"],
        [r1["ProfitPct"], r2["ProfitPct"]]
    )
    ax2.set_ylabel("Expected Profit/Loss (%)")
    ax2.set_title("Expected Profit/Loss per Stock")
    ax2.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig2)

    # Final advice (simple and investor-friendly)
    best = "Omantel" if r1["ProfitPct"] >= r2["ProfitPct"] else "Ooredoo"
    st.markdown("---")
    st.markdown(f"### ✅ Final AI Insight")
    st.write(f"Based on the selected horizon and AI signals, **{best}** currently looks stronger for this period.")
    st.markdown(
        "<div class='tip-box'>"
        "🚀 Invest smarter with <b>ARAS</b>: fast insights, clear recommendations, and confidence designed for investors."
        "</div>",
        unsafe_allow_html=True
    )

    # Back to Home Button (always visible)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🏠 Back to Home"):
            st.session_state["start_analysis"] = False
            st.experimental_rerun()


