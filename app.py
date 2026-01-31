import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# =========================
# Page config
# =========================
st.set_page_config(page_title="ARAS - Smart Portfolio", layout="wide")

# =========================
# Premium UI CSS (Responsive + Professional)
# =========================
st.markdown("""
<style>
.block-container{
  padding-top: 1rem;
  padding-bottom: 90px;
  max-width: 1250px;
  margin: 0 auto;
}
*{ box-sizing: border-box; }

:root{
  --brand:#0B2447;
  --brand2:#1A4D80;
  --card:#FFFFFF;
  --muted:#6B7280;
  --border: rgba(26,77,128,0.18);
  --good:#16A34A;
  --warn:#F59E0B;
  --bad:#DC2626;
}

html, body, [class*="css"]{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}

/* ✅ wrap columns (prevents cut) */
div[data-testid="stHorizontalBlock"]{
  flex-wrap: wrap !important;
  gap: 16px !important;
}
div[data-testid="column"]{
  flex: 1 1 340px !important;
  min-width: 340px !important;
}
@media (max-width: 768px){
  div[data-testid="column"]{ min-width: 100% !important; }
}

/* Top Bar */
.aras-topbar{
  background: linear-gradient(90deg, #D6E6F2, #ECF5FF);
  border:1px solid var(--border);
  border-radius:14px;
  padding: 12px 16px;
  display:flex;
  justify-content: space-between;
  align-items:flex-start;
  gap: 14px;
}
.aras-brand{
  display:flex;
  align-items:flex-start;
  gap: 12px;
  min-width: 0;
}
.aras-logo{
  width:44px;height:44px;border-radius:14px;
  background: linear-gradient(135deg, var(--brand), var(--brand2));
  display:flex;align-items:center;justify-content:center;
  color:white;font-weight:900;letter-spacing:1px;
  flex: 0 0 auto;
}
.aras-title{
  font-weight:900;color:var(--brand);font-size:15px;
  white-space: normal;
  line-height: 1.2;
}
.aras-sub{
  font-weight:700;color:var(--muted);font-size:12px;margin-top:2px;
  white-space: normal;
  line-height: 1.2;
}
.aras-links{
  display:flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-end;
}
.aras-links a{
  text-decoration:none;color:var(--brand2);
  font-weight:800;font-size:13px;
}
.aras-links a:hover{ color:#0D2B4F; }

@media (max-width: 768px){
  .aras-topbar{
    flex-direction: column;
    align-items: flex-start;
    padding: 12px 12px;
  }
  .aras-title{ font-size: 14px; }
  .aras-sub{ font-size: 11px; }
  .aras-links{ justify-content: flex-start; }
}

/* Ticker */
.ticker-wrap{
  margin-top:10px;
  overflow:hidden;
  border-radius:14px;
  background: #0B2447;
  border:1px solid rgba(255,255,255,0.14);
}
.ticker{
  display:inline-block;
  white-space:nowrap;
  padding-left:100%;
  animation: tickerMove 18s linear infinite;
  color:white;
  font-weight:800;
  font-size:13px;
}
.ticker span{ display:inline-block; padding: 10px 22px; opacity:0.98; }
@keyframes tickerMove{
  0% { transform: translate3d(0,0,0); }
  100% { transform: translate3d(-100%,0,0); }
}
@media (max-width: 768px){
  .ticker{ font-size: 12px; }
  .ticker span{ padding: 10px 14px; }
}

/* Hero */
.hero{
  background: radial-gradient(circle at 10% 10%, rgba(26,77,128,0.18), transparent 40%),
              radial-gradient(circle at 90% 20%, rgba(11,36,71,0.20), transparent 45%),
              linear-gradient(135deg, #FFFFFF, #F3F8FF);
  border:1px solid var(--border);
  border-radius:18px;
  padding: 22px 22px;
}
.hero h1{
  color: var(--brand);
  font-size: 42px;
  margin: 0;
  font-weight: 900;
  letter-spacing: 0.3px;
}
.hero p{
  color: #0D2B4F;
  font-size: 15px;
  margin: 10px 0 0 0;
  font-weight: 700;
  opacity: 0.9;
}
.hero .tag{
  display:inline-block;
  margin-top: 12px;
  background: rgba(26,77,128,0.10);
  border: 1px solid rgba(26,77,128,0.20);
  padding: 8px 12px;
  border-radius: 999px;
  color: var(--brand);
  font-weight: 900;
  font-size: 13px;
}
@media (max-width: 768px){
  .hero h1{ font-size: 34px; }
}

/* Cards */
.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 10px 20px rgba(11,36,71,0.04);
  min-height: 96px;
  width: 100%;
  margin-bottom: 12px;
  overflow: hidden;
}
.card .label{
  color: var(--muted);
  font-weight: 900;
  font-size: 12px;
  letter-spacing: 0.3px;
  text-transform: uppercase;
}
.card .value{
  font-size: 22px;
  font-weight: 900;
  color: #111827;
  margin-top: 6px;
}
.card .small{
  color: var(--muted);
  font-weight: 800;
  font-size: 12px;
  margin-top: 6px;
  line-height: 1.3;
}
@media (max-width: 768px){
  .card{ padding: 12px 12px; }
  .card .value{ font-size: 20px; }
}

/* Badges */
.badge{
  display:inline-block;
  padding: 7px 12px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 12px;
  border: 1px solid rgba(0,0,0,0.08);
}
.badge.good{ background: rgba(22,163,74,0.12); color: var(--good); }
.badge.warn{ background: rgba(245,158,11,0.14); color: var(--warn); }
.badge.bad { background: rgba(220,38,38,0.12); color: var(--bad); }

/* Buttons */
div.stButton > button{
  background: linear-gradient(135deg, var(--brand2), var(--brand));
  color: white !important;
  font-size: 16px !important;
  font-weight: 900 !important;
  padding: 10px 16px !important;
  border-radius: 14px !important;
  border: 0 !important;
}
div.stButton > button:hover{ filter: brightness(0.95); }

/* Fixed back button */
.fixed-back{
  position: fixed;
  bottom: 18px;
  right: 18px;
  z-index: 99999;
}
.fixed-back button{
  background: #111827 !important;
  color: white !important;
  font-weight: 900 !important;
  padding: 10px 14px !important;
  border-radius: 999px !important;
  border: 0 !important;
}
.fixed-back button:hover{ background:#000 !important; }

/* ✅ put "Open menu" text near the sidebar arrow */
button[data-testid="collapsedControl"]::after,
button[data-testid="stSidebarCollapsedControl"]::after,
div[data-testid="collapsedControl"] button::after,
div[data-testid="stSidebarCollapsedControl"] button::after,
button[aria-label="Open sidebar"]::after,
button[aria-label="Open sidebar navigation"]::after,
button[title="Open sidebar"]::after,
button[title="Open sidebar navigation"]::after{
  content:" Open menu";
  font-weight: 800;
  font-size: 13px;
  color: #0B2447;
  margin-left: 8px;
  opacity: 0.95;
  white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Top Nav + Ticker
# =========================
st.markdown("""
<div class="aras-topbar">
  <div class="aras-brand">
    <div class="aras-logo">ARAS</div>
    <div>
      <div class="aras-title">ARAS — Smart Investing for the Oman Market</div>
      <div class="aras-sub">AI insights • Clear recommendations • Investor-friendly confidence</div>
    </div>
  </div>
  <div class="aras-links">
    <a href="https://www.msx.om" target="_blank">MSX</a>
    <a href="https://www.omanobserver.om/section/business" target="_blank">Oman News</a>
  </div>
</div>

<div class="ticker-wrap">
  <div class="ticker">
    <span>Monitor MSX with ARAS • Clear signals • Confidence score • Professional summaries</span>
    <span>Tip: Choose horizon 5–20 days for clearer signals</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption("Use the top-left arrow to open the menu and change selections.")

# =========================
# Session state
# =========================
if "start" not in st.session_state:
    st.session_state.start = False

# =========================
# Files
# =========================
FILES = {"Omantel.xlsx": "Omantel.xlsx", "Ooredoo.xlsx": "Ooredoo.xlsx"}

def process_stock_file(file):
    df = pd.read_excel(file)
    df = df[df.iloc[:, 0].astype(str).str.contains(r"\d", regex=True)]
    df = df.iloc[:, :6].copy()
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna().reset_index(drop=True)

def predict_price(df_train, horizon):
    features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI"]

    horizon = int(max(1, horizon))
    horizon = int(min(horizon, max(1, len(df_train) - 2)))

    X = df_train[features]
    y = df_train["Close"].shift(-horizon)

    X = X.iloc[:-horizon]
    y = y.iloc[:-horizon]

    # fallback (more responsive than old one)
    if len(X) < 20 or len(y) < 20:
        last = float(df_train["Close"].iloc[-1])
        ma5 = float(df_train["MA5"].iloc[-1])
        ma10 = float(df_train["MA10"].iloc[-1])
        # small momentum blend
        pred = 0.65 * last + 0.25 * ma5 + 0.10 * ma10
        return float(pred), None, None, None, horizon

    model = RandomForestRegressor(n_estimators=450, random_state=42)
    model.fit(X, y)
    pred = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])
    return pred, model, X, y, horizon

def confidence_score(model, X_test, y_test):
    if model is None or len(X_test) < 10:
        return 45.0
    mae = mean_absolute_error(y_test, model.predict(X_test))
    base = float(y_test.mean()) if float(y_test.mean()) != 0 else 1.0
    error_conf = max(0, 1 - mae / base)

    tree_preds = np.array([tree.predict(X_test.iloc[-1].values.reshape(1, -1))[0] for tree in model.estimators_])
    stability = 1 / (1 + np.std(tree_preds))

    conf = (0.6 * error_conf + 0.4 * stability) * 100
    return float(min(95, max(40, conf)))

def risk_level_from_conf(conf):
    if conf >= 75:
        return "Low Risk", "good"
    if conf >= 55:
        return "Medium Risk", "warn"
    return "High Risk", "bad"

# ✅ أكثر حساسية من السابق (يقلل Hold)
def volatility_threshold_pct(df_ref, horizon_days):
    rets = df_ref["Close"].pct_change().dropna()
    if len(rets) < 20:
        return 0.35
    daily_vol = float(rets.tail(80).std())  # recent vol
    scaled = daily_vol * np.sqrt(max(1, horizon_days)) * 100.0
    # more sensitive band
    thr = scaled * 0.55
    return float(min(2.2, max(0.20, thr)))

def signal_boost_from_indicators(df_last):
    # returns extra bias in percent terms
    rsi = float(df_last["RSI"])
    ma5 = float(df_last["MA5"])
    ma10 = float(df_last["MA10"])
    close = float(df_last["Close"])

    bias = 0.0
    # trend bias
    if ma5 > ma10:
        bias += 0.15
    elif ma5 < ma10:
        bias -= 0.15

    # RSI bias (soft)
    if rsi < 35:
        bias += 0.25
    elif rsi > 65:
        bias -= 0.25

    # price vs MA10
    if close > ma10:
        bias += 0.10
    elif close < ma10:
        bias -= 0.10

    return bias

def recommendation_from_profit(pct, thr, df_last):
    # add indicator bias so it doesn't stay Hold forever
    bias = signal_boost_from_indicators(df_last)
    adj = pct + bias

    if adj > thr:
        return "Buy", "good", adj, bias
    if adj < -thr:
        return "Avoid/Sell", "bad", adj, bias
    return "Hold", "warn", adj, bias

def period_dates(df, preset):
    mx = df["Date"].max().date()
    if preset == "Today":
        return mx, mx
    if preset == "1 Week":
        return (mx - timedelta(days=7)), mx
    if preset == "1 Month":
        return (mx - timedelta(days=30)), mx
    if preset == "1 Year":
        return (mx - timedelta(days=365)), mx
    if preset == "3 Years":
        return (mx - timedelta(days=365 * 3)), mx
    return df["Date"].min().date(), mx

def confidence_progress(conf):
    return max(0.0, min(1.0, conf / 100.0))

# =========================
# HOME
# =========================
if not st.session_state.start:
    st.markdown("""
    <div class="hero">
      <h1>ARAS</h1>
      <p>AI-powered recommendations that help you save time, reduce noise, and invest smarter in the Oman market.</p>
      <div class="tag">Clear Signals • Smart Charts • Confidence Score</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    colA, colB = st.columns([1.2, 1])

    with colA:
        st.markdown("""
        <div class="card">
          <div class="label">What you get</div>
          <div class="value">Smart, clear decisions</div>
          <div class="small">Pick a company and horizon → ARAS generates a clean report with charts and a confidence score.</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="card">
          <div class="label">Designed for Oman investors</div>
          <div class="value">Fast and professional</div>
          <div class="small">Simple indicators. Professional output that fits a tech product.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("Start Analysis"):
        st.session_state.start = True
        st.rerun()

# =========================
# DASHBOARD
# =========================
if st.session_state.start:
    st.markdown("## Investor Dashboard")

    # Sidebar selections (kept!)
    st.sidebar.markdown("### Selections")
    company = st.sidebar.selectbox("Select Company", list(FILES.keys()))
    df_full = process_stock_file(FILES[company])

    # History view period (for charts only)
    preset = st.sidebar.radio("History View", ["1 Week", "1 Month", "1 Year", "3 Years", "Custom (Calendar)"], index=1)

    min_d = df_full["Date"].min().date()
    max_d = df_full["Date"].max().date()

    if preset != "Custom (Calendar)":
        start_d, end_d = period_dates(df_full, preset)
        start_d = max(start_d, min_d)
        end_d = min(end_d, max_d)
        st.sidebar.caption(f"History range: {start_d} → {end_d}")
    else:
        picked = st.sidebar.date_input(
            "Pick Start & End",
            value=[max_d - timedelta(days=30), max_d],
            min_value=min_d,
            max_value=max_d
        )
        if isinstance(picked, datetime):
            start_d, end_d = picked.date(), picked.date()
        else:
            start_d, end_d = picked[0], picked[1]
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    # ✅ NEW: Forecast horizon separated (fixes Hold issue)
    horizon = st.sidebar.select_slider(
        "Forecast Horizon (days)",
        options=[1, 3, 5, 10, 20, 30, 60],
        value=10
    )

    st.sidebar.info("Tip: Horizon 5–20 days often shows clearer signals than 1 day.")

    # Training window: last ~260 rows (stronger model + less Hold)
    df_train = df_full.tail(max(260, horizon + 80)).copy()
    pred, model, X, y, horizon = predict_price(df_train, horizon)

    # Current = latest close (real investor view)
    current = float(df_full.iloc[-1]["Close"])
    current_date = df_full.iloc[-1]["Date"]

    profit_pct = (pred - current) / current * 100 if current != 0 else 0.0
    thr = volatility_threshold_pct(df_train, horizon)

    # Recommendation with indicator bias
    df_last = df_train.iloc[-1]
    rec, rec_tone, adj_pct, bias = recommendation_from_profit(profit_pct, thr, df_last)

    # Confidence
    if model is None:
        conf = 45.0
        mode = "Quick Insight"
    else:
        if len(X) < 20 or len(y) < 20:
            conf = 50.0
            mode = "AI Model (Limited Data)"
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
            conf = confidence_score(model, Xte, yte)
            mode = "AI Model"

    risk_text, risk_tone = risk_level_from_conf(conf)

    # Cards
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    c1.markdown(f"""
    <div class="card">
      <div class="label">Current Price</div>
      <div class="value">{current:.3f} OMR</div>
      <div class="small">Date: {current_date.date()}</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="card">
      <div class="label">Predicted Price</div>
      <div class="value">{pred:.3f} OMR</div>
      <div class="small">Horizon: {horizon} days • Mode: {mode}</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="card">
      <div class="label">Expected Change</div>
      <div class="value">{profit_pct:.2f}%</div>
      <div class="small">Threshold: ±{thr:.2f}% • Indicator bias: {bias:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="card">
      <div class="label">Recommendation</div>
      <div class="value">{rec}</div>
      <div class="small">Adjusted signal: {adj_pct:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence + badges
    colx, coly = st.columns([1.4, 1])
    with colx:
        st.markdown(f"**AI Confidence:** {conf:.1f}%")
        st.progress(confidence_progress(conf))
        st.caption("Confidence reflects stability and historical error. With limited data windows, a simplified estimate may apply.")
    with coly:
        st.markdown(f"<span class='badge {risk_tone}'>{risk_text}</span>", unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f"<span class='badge {rec_tone}'>{rec}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # History chart (selected period)
    st.subheader("Price Chart (History View)")
    df_hist = df_full[(df_full["Date"].dt.date >= start_d) & (df_full["Date"].dt.date <= end_d)].copy()
    if len(df_hist) < 5:
        df_hist = df_full.tail(120).copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_hist["Date"], df_hist["Close"], label="Close")
    ax.set_title(f"{company} | History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (OMR)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Actual vs Predicted (Latest Forecast)")
    future_date = current_date + pd.Timedelta(days=horizon)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df_full["Date"], df_full["Close"], label="Actual Price")
    ax2.scatter(future_date, pred, s=90, label="Predicted Price")
    ax2.set_title(f"{company} | Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (OMR)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

    # Back button
    st.markdown('<div class="fixed-back">', unsafe_allow_html=True)
    if st.button("Back to Home"):
        st.session_state.start = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

