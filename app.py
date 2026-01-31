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
st.set_page_config(
    page_title="ARAS - Smart Portfolio",
    layout="wide",
    initial_sidebar_state="expanded"  # ✅ يظهر المنيو على الكمبيوتر تلقائياً
)

# =========================
# Premium UI CSS (Official + Responsive)
# =========================
st.markdown("""
<style>
/* General layout */
.block-container{
  padding-top: 0.8rem;
  padding-bottom: 90px;
  max-width: 1200px;
}

/* Make everything behave nicely on small screens */
*{ box-sizing: border-box; }
img, svg { max-width: 100%; height: auto; }
[data-testid="stHorizontalBlock"]{ gap: 12px; }
[data-testid="stVerticalBlock"]{ gap: 12px; }

/* Brand tokens */
:root{
  --brand:#0B2447;
  --brand2:#1A4D80;
  --bg:#F6F9FC;
  --card:#FFFFFF;
  --muted:#6B7280;
  --border: rgba(26,77,128,0.18);
  --good:#16A34A;
  --warn:#F59E0B;
  --bad:#DC2626;
}

html, body, [class*="css"]{
  font-family: Arial, sans-serif;
}

/* ✅ Sidebar toggle label next to the arrow (no UI breaking) */
[data-testid="collapsedControl"]{
  align-items: center;
}
[data-testid="collapsedControl"]::after{
  content: " Open menu";
  font-weight: 800;
  font-size: 13px;
  color: var(--brand);
  margin-left: 6px;
  opacity: 0.95;
}
[data-testid="stSidebarCollapsedControl"]::after{
  content: " Open menu";
  font-weight: 800;
  font-size: 13px;
  color: var(--brand);
  margin-left: 6px;
  opacity: 0.95;
}

/* Top Bar */
.aras-topbar{
  background: linear-gradient(90deg, #D6E6F2, #ECF5FF);
  border:1px solid var(--border);
  border-radius:14px;
  padding: 10px 16px;
  display:flex;
  justify-content: space-between;
  align-items:flex-start;
  gap: 12px;
  flex-wrap: wrap;
}
.aras-brand{
  display:flex;
  align-items:center;
  gap: 12px;
  min-width: 0;
  flex: 1 1 520px;
}
.aras-logo{
  width:44px;height:44px;border-radius:14px;
  background: linear-gradient(135deg, var(--brand), var(--brand2));
  display:flex;align-items:center;justify-content:center;
  color:white;font-weight:900;letter-spacing:2px;
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
  flex: 0 1 auto;
}
.aras-links a{
  text-decoration:none;color:var(--brand2);
  font-weight:800;font-size:13px;
}
.aras-links a:hover{ color:#0D2B4F; }

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

/* Hero */
.hero{
  background: radial-gradient(circle at 10% 10%, rgba(26,77,128,0.16), transparent 40%),
              radial-gradient(circle at 90% 20%, rgba(11,36,71,0.18), transparent 45%),
              linear-gradient(135deg, #FFFFFF, #F3F8FF);
  border:1px solid var(--border);
  border-radius:18px;
  padding: 22px 22px;
}
.hero h1{
  color: var(--brand);
  font-size: 44px;
  margin: 0;
  font-weight: 900;
  letter-spacing: 0.2px;
}
.hero p{
  color: #0D2B4F;
  font-size: 15px;
  margin: 10px 0 0 0;
  font-weight: 700;
  opacity: 0.92;
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

/* Cards */
.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 10px 20px rgba(11,36,71,0.04);
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
  word-break: break-word;
}
.card .small{
  color: var(--muted);
  font-weight: 800;
  font-size: 12px;
  margin-top: 6px;
  word-break: break-word;
}

/* Selections card slightly smaller */
.selections-card{
  padding: 10px 12px !important;
}

/* Responsive metrics grid (fix overlapping/cutting) */
.metrics-grid{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 12px;
}
@media (max-width: 520px){
  .metrics-grid{ grid-template-columns: 1fr; }
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

/* Buttons (official) */
div.stButton > button{
  background: linear-gradient(135deg, var(--brand2), var(--brand));
  color: white !important;
  font-size: 16px !important;
  font-weight: 900 !important;
  padding: 10px 16px !important;
  border-radius: 12px !important;
  border: 0 !important;
}
div.stButton > button:hover{ filter: brightness(0.96); }

/* Fixed back button */
.fixed-back{
  position: fixed;
  bottom: 16px;
  right: 16px;
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
</style>
""", unsafe_allow_html=True)

# =========================
# Top Nav + Ticker (Marketing but official)
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
    <span>ARAS provides clear signals, confidence scoring, and simple investor-ready insights.</span>
    <span>Tip: Short ranges are great for quick checks; longer ranges improve confidence and stability.</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Session state
# =========================
if "start" not in st.session_state:
    st.session_state.start = False

# =========================
# Data loading + features
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

def predict_price(df, horizon):
    features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI"]

    horizon = int(max(1, horizon))
    horizon = int(min(horizon, max(1, len(df) - 2)))

    X = df[features]
    y = df["Close"].shift(-horizon)

    X = X.iloc[:-horizon]
    y = y.iloc[:-horizon]

    # Quick fallback for tiny data windows
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

def volatility_threshold_pct(df_for_thr, floor_pct=0.6):
    """
    Dynamic threshold so Recommendation isn't always Hold.
    Uses recent volatility (std of daily returns) -> % threshold.
    """
    if df_for_thr is None or len(df_for_thr) < 15:
        return float(floor_pct)

    r = df_for_thr["Close"].pct_change().dropna()
    if len(r) < 10:
        return float(floor_pct)

    vol = float(r.tail(30).std())  # daily volatility
    thr = max(floor_pct, 0.75 * (vol * 100.0))  # % threshold
    return float(min(3.5, thr))  # cap to keep reasonable

def recommendation_from_profit(pct, threshold):
    if pct > threshold:
        return "Buy", "good"
    if pct < -threshold:
        return "Sell / Avoid", "bad"
    return "Hold", "warn"

# =========================
# HOME
# =========================
if not st.session_state.start:
    st.markdown("""
    <div class="hero">
      <h1>ARAS</h1>
      <p>AI-powered recommendations to help you save time, reduce noise, and invest smarter in the Oman market.</p>
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
          <div class="small">Choose a company and a time period; ARAS generates a clear report with charts and confidence.</div>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class="card">
          <div class="label">Built for investors</div>
          <div class="value">Fast and simple</div>
          <div class="small">Clean outputs, professional layout, and practical insights without complexity.</div>
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

    # ✅ Sidebar selections (رجّعناه رسمي)
    with st.sidebar:
        st.markdown("<div class='card selections-card'><div class='label'>Selections</div>", unsafe_allow_html=True)

        company = st.selectbox("Select Company", list(FILES.keys()))
        df_full = process_stock_file(FILES[company])

        preset = st.radio("Select Period", ["Today", "1 Week", "1 Month", "1 Year", "3 Years", "Custom (Calendar)"])

        min_d = df_full["Date"].min().date()
        max_d = df_full["Date"].max().date()

        if preset != "Custom (Calendar)":
            start_d, end_d = period_dates(df_full, preset)
            start_d = max(start_d, min_d)
            end_d = min(end_d, max_d)
            st.caption(f"Selected range: {start_d} → {end_d}")
        else:
            picked = st.date_input(
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

        st.info("Tip: Short ranges are fine for quick checks. Longer ranges improve stability and confidence.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Build window based on selected dates
    df_win = df_full[(df_full["Date"].dt.date >= start_d) & (df_full["Date"].dt.date <= end_d)].copy()

    # If user picks very short ranges, still compute but ensure minimum window for features
    if len(df_win) < 25:
        df_win = df_full.tail(max(120, len(df_win))).copy()

    # Horizon (days)
    horizon = max(1, (pd.to_datetime(end_d) - pd.to_datetime(start_d)).days)
    pred, model, X, y, horizon = predict_price(df_win, horizon)

    # Use selected-period close and latest close
    selected_close = float(df_win.iloc[-1]["Close"])
    latest_close = float(df_full.iloc[-1]["Close"])

    # Profit vs latest close (can change to selected_close if you prefer)
    profit_pct = (pred - latest_close) / latest_close * 100 if latest_close != 0 else 0.0

    # Smarter threshold based on volatility
    thr = volatility_threshold_pct(df_full, floor_pct=0.6)
    rec, rec_tone = recommendation_from_profit(profit_pct, thr)

    # Confidence
    if model is None:
        conf = 45.0
        mode = "Quick Insight"
    else:
        if X is None or y is None or len(X) < 10 or len(y) < 10:
            conf = 50.0
            mode = "AI (Limited Data)"
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
            conf = confidence_score(model, Xte, yte)
            mode = "AI Model"

    risk_text, risk_tone = risk_level_from_conf(conf)

    # =========================
    # Main content
    # =========================

    # ✅ Responsive metrics (no cutting / no overlap)
    st.markdown(f"""
    <div class="metrics-grid">
      <div class="card">
        <div class="label">Selected Period Close</div>
        <div class="value">{selected_close:.3f} OMR</div>
        <div class="small">Price at end of chosen range</div>
      </div>

      <div class="card">
        <div class="label">Predicted Price</div>
        <div class="value">{pred:.3f} OMR</div>
        <div class="small">Horizon: {horizon} day(s)</div>
      </div>

      <div class="card">
        <div class="label">Expected Change</div>
        <div class="value">{profit_pct:.2f}%</div>
        <div class="small">Signal threshold: ±{thr:.2f}% (volatility-based)</div>
      </div>

      <div class="card">
        <div class="label">Recommendation</div>
        <div class="value">{rec}</div>
        <div class="small">Mode: {mode}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Confidence + badges
    colx, coly = st.columns([1.35, 1])
    with colx:
        st.markdown(f"**AI Confidence:** {conf:.1f}%")
        st.progress(confidence_progress(conf))
        st.caption("Confidence reflects stability and historical error. In limited data windows, a simplified estimate is used.")
    with coly:
        st.markdown(f"<span class='badge {risk_tone}'>{risk_text}</span>", unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f"<span class='badge {rec_tone}'>{rec}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Chart
    st.subheader("Price Chart (Actual vs Predicted)")
    future_date = df_full.iloc[-1]["Date"] + pd.Timedelta(days=horizon)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_full["Date"], df_full["Close"], label="Actual Price")
    ax.scatter(future_date, pred, s=90, label="Predicted Price")
    ax.set_title(f"{company} | Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (OMR)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Compare section
    st.subheader("Comparison (Omantel vs Ooredoo)")
    df_om = process_stock_file(FILES["Omantel.xlsx"])
    df_oo = process_stock_file(FILES["Ooredoo.xlsx"])

    def quick_analyze(df_comp, name):
        df_cut = df_comp.tail(max(140, horizon + 60)).copy()
        p, m, Xc, yc, _ = predict_price(df_cut, horizon)
        last = float(df_comp.iloc[-1]["Close"])
        pct = (p - last) / last * 100 if last != 0 else 0.0

        local_thr = volatility_threshold_pct(df_comp, floor_pct=0.6)
        rec_local, _tone = recommendation_from_profit(pct, local_thr)

        if m is None or Xc is None or len(Xc) < 10:
            cf = 45.0
        else:
            Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, shuffle=False)
            cf = confidence_score(m, Xte, yte)

        trend = "Up" if pct > 0 else "Down"
        return {"Stock": name, "Expected %": pct, "Threshold %": local_thr, "Recommendation": rec_local, "Confidence %": cf, "Trend": trend}

    r1 = quick_analyze(df_om, "Omantel")
    r2 = quick_analyze(df_oo, "Ooredoo")

    cmp_df = pd.DataFrame([r1, r2])
    st.dataframe(cmp_df, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(["Omantel", "Ooredoo"], [r1["Expected %"], r2["Expected %"]])
    ax2.set_title("Expected Profit/Loss (%)")
    ax2.set_ylabel("%")
    ax2.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig2)

    st.success("ARAS provides structured, investor-ready insights with clarity and confidence.")

    # Fixed Back to Home
    st.markdown('<div class="fixed-back">', unsafe_allow_html=True)
    if st.button("Back to Home"):
        st.session_state.start = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
