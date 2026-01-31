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
# Corporate UI CSS (Responsive + No Emojis Look)
# =========================
st.markdown("""
<style>
/* Prevent horizontal overflow */
html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }

/* More bottom space (mobile browser bars) */
.block-container { padding-top: 0.9rem; padding-bottom: 180px !important; }

/* Typography: corporate */
:root{
  --brand:#0B2447;
  --brand2:#1A4D80;
  --bg:#F6F9FC;
  --card:#FFFFFF;
  --text:#0F172A;
  --muted:#64748B;
  --border: rgba(15, 23, 42, 0.10);
  --shadow: 0 10px 22px rgba(2, 8, 23, 0.05);
  --good:#16A34A;
  --warn:#D97706;
  --bad:#DC2626;
}

html, body, [class*="css"]{
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Noto Sans", "Helvetica Neue", sans-serif;
  color: var(--text);
}

/* Top Bar */
.aras-topbar{
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
  display:flex;
  justify-content: space-between;
  align-items:center;
  gap: 14px;
  box-shadow: var(--shadow);
}
.aras-brand{ display:flex; align-items:center; gap: 12px; min-width: 0; }
.aras-logo{
  width:42px;height:42px;border-radius:12px;
  background: linear-gradient(135deg, var(--brand), var(--brand2));
  display:flex;align-items:center;justify-content:center;
  color:white;font-weight:800;letter-spacing:1px;
  flex: 0 0 auto;
}
.aras-title{
  font-weight:800;
  color: var(--brand);
  font-size: 14px;
  line-height: 1.25;
}
.aras-sub{
  font-weight:600;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.25;
  margin-top: 2px;
}
.aras-links{
  display:flex; flex-wrap: wrap; gap: 10px;
  justify-content: flex-end; align-items:center;
}
.aras-links a{
  text-decoration:none;
  color: var(--brand2);
  font-weight:700;
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 10px;
  border: 1px solid rgba(26,77,128,0.18);
  background: rgba(26,77,128,0.04);
}
.aras-links a:hover{ background: rgba(26,77,128,0.08); }

/* Mobile top bar */
@media (max-width: 768px){
  .aras-topbar{
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  .aras-links{ justify-content: flex-start; }
}

/* Helper banner (to guide users to open sidebar) */
.helper{
  margin-top: 10px;
  background: rgba(11,36,71,0.05);
  border: 1px solid rgba(11,36,71,0.10);
  border-radius: 12px;
  padding: 10px 12px;
  color: var(--brand);
  font-weight: 700;
  font-size: 12px;
}

/* Hero */
.hero{
  margin-top: 12px;
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: var(--shadow);
}
.hero h1{
  color: var(--brand);
  font-size: 40px;
  margin: 0;
  font-weight: 900;
  letter-spacing: 0.3px;
}
.hero p{
  color: var(--muted);
  font-size: 14px;
  margin: 10px 0 0 0;
  font-weight: 600;
  line-height: 1.55;
}

/* Cards */
.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px;
  box-shadow: var(--shadow);

  height: auto !important;
  min-height: 0 !important;
  overflow: visible !important;
}
.card .label{
  color: var(--muted);
  font-weight: 800;
  font-size: 11px;
  letter-spacing: 0.4px;
  text-transform: uppercase;
}
.card .value{
  font-size: 20px;
  font-weight: 900;
  color: var(--text);
  margin-top: 6px;
  white-space: normal;
  word-break: break-word;
}
.card .small{
  color: var(--muted);
  font-weight: 600;
  font-size: 12px;
  margin-top: 6px;
  white-space: normal;
  word-break: break-word;
  line-height: 1.5;
}

/* Spacing between cards (especially mobile) */
.cards-grid{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
@media (max-width: 1024px){
  .cards-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 768px){
  .cards-grid{
    grid-template-columns: 1fr;
    gap: 16px; /* bigger spacing so they don't feel stuck together */
  }
}

/* Badges (clean) */
.badge{
  display:inline-block;
  padding: 7px 10px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 12px;
  border: 1px solid rgba(2,8,23,0.08);
}
.badge.good{ background: rgba(22,163,74,0.10); color: var(--good); }
.badge.warn{ background: rgba(217,119,6,0.12); color: var(--warn); }
.badge.bad { background: rgba(220,38,38,0.10); color: var(--bad); }

/* Buttons */
div.stButton > button{
  background: linear-gradient(135deg, var(--brand2), var(--brand));
  color: white !important;
  font-size: 14px !important;
  font-weight: 800 !important;
  padding: 10px 14px !important;
  border-radius: 12px !important;
  border: 0 !important;
}

/* Section titles */
.section-title{
  font-size: 18px;
  font-weight: 900;
  color: var(--brand);
  margin: 10px 0 6px 0;
}

/* Sidebar styling (subtle) */
section[data-testid="stSidebar"]{
  border-right: 1px solid rgba(2,8,23,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================
# Top Bar
# =========================
st.markdown("""
<div class="aras-topbar">
  <div class="aras-brand">
    <div class="aras-logo">ARAS</div>
    <div>
      <div class="aras-title">ARAS Smart Portfolio</div>
      <div class="aras-sub">AI insights, clear recommendations, investor-friendly confidence</div>
    </div>
  </div>
  <div class="aras-links">
    <a href="https://www.msx.om" target="_blank">MSX</a>
    <a href="https://www.omanobserver.om/section/business" target="_blank">Oman Business News</a>
  </div>
</div>

<div class="helper">
  To change company and time period, open the Selections panel using the menu button (top-left) or use the sidebar on desktop.
</div>
""", unsafe_allow_html=True)

# =========================
# Session state
# =========================
if "start" not in st.session_state:
    st.session_state.start = False

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

def predict_price(df, horizon):
    features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI"]

    horizon = int(max(1, horizon))
    horizon = int(min(horizon, max(1, len(df) - 2)))

    X = df[features]
    y = df["Close"].shift(-horizon)

    X = X.iloc[:-horizon]
    y = y.iloc[:-horizon]

    if len(X) < 12 or len(y) < 12:
        last = float(df["Close"].iloc[-1])
        ma5 = float(df["MA5"].iloc[-1]) if "MA5" in df.columns else last
        pred = (0.75 * last) + (0.25 * ma5)
        return float(pred), None, None, None, horizon

    model = RandomForestRegressor(n_estimators=350, random_state=42)
    model.fit(X, y)
    pred = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])
    return pred, model, X, y, horizon

def confidence_score(model, X_test, y_test):
    if model is None or len(X_test) < 5:
        return 45.0
    mae = mean_absolute_error(y_test, model.predict(X_test))
    base = float(y_test.mean()) if float(y_test.mean()) != 0 else 1.0
    error_conf = max(0, 1 - mae / base)

    tree_preds = np.array([tree.predict(X_test.iloc[-1].values.reshape(1, -1))[0] for tree in model.estimators_])
    stability = 1 / (1 + np.std(tree_preds))

    conf = (0.6 * error_conf + 0.4 * stability) * 100
    return float(min(95, max(40, conf)))

def dynamic_threshold_pct(df):
    rets = df["Close"].pct_change().dropna()
    if len(rets) < 20:
        return 0.8
    vol = float(rets.tail(60).std() * 100)
    thr = max(0.6, min(3.0, vol * 0.9))
    return float(thr)

def recommendation_from_profit(pct, thr):
    if pct > thr:
        return "Buy", "good"
    if pct < -thr:
        return "Avoid", "bad"
    return "Hold", "warn"

def risk_level_from_conf(conf):
    if conf >= 75:
        return "Low Risk", "good"
    if conf >= 55:
        return "Medium Risk", "warn"
    return "High Risk", "bad"

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
      <p>AI-powered portfolio insights designed to support investment decisions in the Oman market with clarity and confidence.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
          <div class="label">What you get</div>
          <div class="value">Clear, decision-ready signals</div>
          <div class="small">Select a company and time period to generate an investor-focused report with charts and confidence.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
          <div class="label">Built for professionals</div>
          <div class="value">Fast and structured</div>
          <div class="small">A clean interface that presents key results first, with details available below.</div>
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
    st.markdown("<div class='section-title'>Investor Dashboard</div>", unsafe_allow_html=True)

    # Sidebar selections (works best on desktop; mobile opens from menu button)
    st.sidebar.markdown("### Selections")
    company = st.sidebar.selectbox("Company", list(FILES.keys()))
    df_full = process_stock_file(FILES[company])

    preset = st.sidebar.radio("Time Period", ["Today", "1 Week", "1 Month", "1 Year", "3 Years", "Custom (Calendar)"])

    min_d = df_full["Date"].min().date()
    max_d = df_full["Date"].max().date()

    if preset != "Custom (Calendar)":
        start_d, end_d = period_dates(df_full, preset)
        start_d = max(start_d, min_d)
        end_d = min(end_d, max_d)
        st.sidebar.caption(f"Selected range: {start_d} to {end_d}")
    else:
        picked = st.sidebar.date_input(
            "Start and End Date",
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
        st.sidebar.caption(f"Selected range: {start_d} to {end_d}")

    st.sidebar.info("Tip: Longer periods typically improve confidence by providing more historical context.")

    if st.sidebar.button("Back to Home"):
        st.session_state.start = False
        st.rerun()

    # Build window
    df_win = df_full[(df_full["Date"].dt.date >= start_d) & (df_full["Date"].dt.date <= end_d)].copy()
    if len(df_win) < 20:
        df_win = df_full.tail(160).copy()

    # Horizon & prediction
    horizon = max(1, (pd.to_datetime(end_d) - pd.to_datetime(start_d)).days)
    pred, model, X, y, horizon = predict_price(df_win, horizon)

    base_close = float(df_win["Close"].iloc[-1])
    profit_pct = (pred - base_close) / base_close * 100 if base_close != 0 else 0.0

    thr = dynamic_threshold_pct(df_win)
    rec, rec_tone = recommendation_from_profit(profit_pct, thr)

    if model is None:
        conf = 45.0
        mode = "Quick Insight"
    else:
        if len(X) < 10 or len(y) < 10:
            conf = 50.0
            mode = "Model (Limited Data)"
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
            conf = confidence_score(model, Xte, yte)
            mode = "Model"

    risk_text, risk_tone = risk_level_from_conf(conf)

    # Cards grid (responsive)
    st.markdown("<div class='cards-grid'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
      <div class="label">Selected Period Close</div>
      <div class="value">{base_close:.3f} OMR</div>
      <div class="small">Price at the end of the chosen period</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
      <div class="label">Predicted Price</div>
      <div class="value">{pred:.3f} OMR</div>
      <div class="small">Forecast horizon: {horizon} day(s)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
      <div class="label">Expected Change</div>
      <div class="value">{profit_pct:.2f}%</div>
      <div class="small">Signal threshold: ±{thr:.2f}% (volatility-based)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
      <div class="label">Recommendation</div>
      <div class="value">{rec}</div>
      <div class="small">Mode: {mode}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown(f"**Model Confidence:** {conf:.1f}%")
        st.progress(confidence_progress(conf))
        st.caption("Confidence reflects model stability and historical error (or a simplified estimate when data is limited).")
    with right:
        st.markdown(f"<span class='badge {risk_tone}'>{risk_text}</span>", unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f"<span class='badge {rec_tone}'>{rec}</span>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div class='section-title'>Price Chart</div>", unsafe_allow_html=True)
    future_date = df_win.iloc[-1]["Date"] + pd.Timedelta(days=horizon)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_full["Date"], df_full["Close"], label="Actual")
    ax.scatter(future_date, pred, s=90, label="Predicted")
    ax.set_title(f"{company} | Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (OMR)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
