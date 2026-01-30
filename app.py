import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="ARAS - Smart Portfolio", layout="wide")

# ---- Top Navigation Bar + Moving Marquee (Official, soft blue) ----
st.markdown("""
<style>
.top-bar {
    background-color: #D6E6F2;
    padding: 10px 25px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    border-bottom: 1px solid #A9CFE7;
    font-family: Arial, sans-serif;
    font-size: 14px;
}
.top-bar-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.top-bar a {
    text-decoration: none;
    color: #1A4D80;
    font-weight: 600;
    margin-left: 15px;
}
.top-bar a:hover { color: #0D2B4F; }
.top-title {
    font-weight: 800;
    color: #123B66;
    font-size: 15px;
}

/* Moving ticker (continuous) */
.ticker-wrap{
    width:100%;
    overflow:hidden;
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(26,77,128,0.25);
    border-radius: 10px;
    padding: 6px 0;
}
.ticker{
    display:inline-block;
    white-space:nowrap;
    padding-left:100%;
    animation: tickerMove 18s linear infinite;
    font-weight:700;
    color:#1A4D80;
    font-size:14px;
}
@keyframes tickerMove{
    0% { transform: translateX(0%); }
    100% { transform: translateX(-200%); }
}
.ticker span{ margin-right: 45px; }
</style>

<div class="top-bar">
  <div class="top-bar-row">
      <div class="top-title">📊 ARAS – Smart Portfolio</div>
      <div>
          <a href="https://www.msx.om" target="_blank">📰 Muscat Stock Exchange</a>
          <a href="https://www.omanobserver.om/section/business" target="_blank">📈 Oman Market News</a>
      </div>
  </div>
  <div class="ticker-wrap">
      <div class="ticker">
          <span>🚀 ARAS Tip: Choose shorter periods for sharper, faster, and higher-confidence insights.</span>
          <span>📊 Compare Omantel & Ooredoo in seconds — save time and stay ahead.</span>
          <span>🤖 AI-powered signals designed to support smarter decisions in Oman’s market.</span>
          <span>✅ Pro Tip: Shorter ranges help the model learn patterns better and reduce noise.</span>
      </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- Initialize session_state ----
if 'start_analysis' not in st.session_state:
    st.session_state['start_analysis'] = False

# ---- Welcome Page ----
if not st.session_state['start_analysis']:
    st.markdown("<h1 style='text-align: center; color: #1A4D80; font-size:50px;'>Welcome to ARAS</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #1A4D80; font-size:28px;'>An Intelligent Investment Recommendation System for the Oman Stock Market Using Artificial Intelligence</h5>", unsafe_allow_html=True)
    st.markdown("---")

    ads = [
        ("📈 Predict stock prices before the market moves!", "#D6E6F2"),
        ("🤖 Powered confidence scores for Omantel & Ooredoo!", "#D6E6F2"),
        ("📊 Compare top stocks in seconds!", "#D6E6F2"),
        ("💡 Make smarter investment decisions today!", "#D6E6F2")
    ]
    for text, color in ads:
        st.markdown(f"""
        <div style='background-color:{color}; padding:25px; border-radius:15px; margin-bottom:15px;
                    color:#000; text-align:center; font-size:24px; font-weight:bold;'>{text}</div>
        """, unsafe_allow_html=True)

    if st.button("🚀 Start Analysis"):
        st.session_state['start_analysis'] = True

# ---- Main Analysis Page ----
if st.session_state['start_analysis']:
    st.success("ARAS Loaded! Stock analysis starts below...")

    # ---- Preloaded stock files ----
    files_dict = {
        "Omantel.xlsx": "Omantel.xlsx",
        "Ooredoo.xlsx": "Ooredoo.xlsx"
    }

    # Stock selection (investor chooses company)
    stock_choice = st.selectbox("Select Company", list(files_dict.keys()))

    # ---- Helper functions ----
    def process_stock_file(file):
        df = pd.read_excel(file)
        df = df[df.iloc[:, 0].astype(str).str.contains(r"\d", regex=True)]
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
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        return df.dropna()

    def predict_price(df, horizon):
        features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI"]
        X = df[features]
        y = df["Close"].shift(-horizon)

        X = X.iloc[:-horizon]
        y = y.iloc[:-horizon]

        if len(X) < 20 or len(y) < 20:
            return np.nan, None, None, None

        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        predicted = model.predict(X.iloc[-1].values.reshape(1, -1))[0]
        return predicted, model, X, y

    def confidence_score(model, X_test, y_test):
        mae = mean_absolute_error(y_test, model.predict(X_test))
        base = y_test.mean() if y_test.mean() != 0 else 1.0
        error_conf = max(0, 1 - mae / base)

        tree_preds = np.array([tree.predict(X_test.iloc[-1].values.reshape(1, -1))[0] for tree in model.estimators_])
        stability = 1 / (1 + np.std(tree_preds))

        confidence = (0.6 * error_conf + 0.4 * stability) * 100
        return float(min(95, max(40, confidence)))

    def compare_stocks(name1, df1, name2, df2, horizon):
        def analyze(df, name):
            pred, model, X, y = predict_price(df, horizon)
            if model is None:
                return {"Name": name, "Last Close": np.nan, "Predicted": np.nan, "Profit %": np.nan,
                        "Trend": "N/A", "Confidence": np.nan}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            conf = confidence_score(model, X_test, y_test)

            last = df.iloc[-1]
            profit_pct = (pred - last["Close"]) / last["Close"] * 100
            trend = "Up ✅" if profit_pct > 0 else "Down ❌"

            return {
                "Name": name,
                "Last Close": float(last["Close"]),
                "Predicted": float(pred),
                "Profit %": float(profit_pct),
                "Trend": trend,
                "Confidence": conf
            }

        return analyze(df1, name1), analyze(df2, name2)

    # ---- Load selected stock data ----
    df_full = process_stock_file(files_dict[stock_choice])

    # ---- Investor chooses period (no auto default long range) ----
    st.subheader("Choose Time Period (Investor Selection)")

    period = st.radio(
        "Select analysis period",
        ["Today", "1 Week", "1 Month", "1 Year", "3 Years", "Custom (Calendar)"],
        horizontal=True
    )

    min_date = df_full["Date"].min().date()
    max_date = df_full["Date"].max().date()

    if period != "Custom (Calendar)":
        days_map = {
            "Today": 5,
            "1 Week": 7,
            "1 Month": 30,
            "1 Year": 365,
            "3 Years": 365 * 3
        }
        days = days_map[period]
        end_date = max_date
        start_date = max(min_date, end_date - timedelta(days=days))
    else:
        picked = st.date_input(
            "Select Start and End Dates",
            value=(),  # investor must choose
            min_value=min_date,
            max_value=max_date
        )
        if not picked or len(picked) != 2:
            st.info("Please choose a start and end date to continue.")
            st.stop()
        start_date, end_date = picked
        if end_date < start_date:
            start_date, end_date = end_date, start_date

    # ---- Filter by chosen period ----
    df = df_full[(df_full["Date"].dt.date >= start_date) & (df_full["Date"].dt.date <= end_date)].copy()
    df = df.sort_values("Date").dropna()

    # If too short, ask to extend (marketing note)
    if len(df) < 60:
        st.info("Smart Tip: Shorter ranges are great, but make sure the selected period includes enough data (try a longer range) for higher-confidence insights.")
        st.stop()

    # Horizon = number of days between start/end (kept safe)
    horizon_days = max(1, (end_date - start_date).days)
    horizon_days = min(horizon_days, len(df) - 1)

    # Soft marketing note only if horizon still very large
    if horizon_days > 365:
        st.info("Smart Tip: Shorter date ranges help ARAS deliver sharper, faster, and higher-confidence insights — try selecting a shorter period to get the best results.")

    # ---- Main Prediction ----
    predicted_price, model, X, y = predict_price(df, horizon_days)
    if model is None:
        st.info("Smart Tip: Choose a slightly longer period so ARAS can learn patterns and generate a clearer prediction.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    confidence = confidence_score(model, X_test, y_test)

    current_price = float(df.iloc[-1]["Close"])
    profit_pct = (predicted_price - current_price) / current_price * 100
    future_date = df.iloc[-1]["Date"] + pd.Timedelta(days=horizon_days)

    if profit_pct > 1:
        recommendation, rec_color = "Buy 📈", "#00AA00"
    elif profit_pct < -1:
        recommendation, rec_color = "Avoid/Sell 📉", "#FF0000"
    else:
        recommendation, rec_color = "Hold ⚪", "#FFA500"

    # ---- Display report ----
    st.subheader(f"Stock Report: {stock_choice}")
    st.write(f"**Selected Period:** {start_date} → {end_date}")
    st.write(f"**Current Price:** {current_price:.3f} OMR")
    st.write(f"**Predicted Price ({horizon_days} days):** {predicted_price:.3f} OMR")
    st.write(f"**Profit Expectation:** {profit_pct:.2f}%")
    st.write(f"**Confidence Score:** {confidence:.1f}%")
    st.markdown(f"**Recommendation:** <span style='color:{rec_color}; font-weight:800; font-size:18px;'>{recommendation}</span>", unsafe_allow_html=True)

    # ---- Actual vs Predicted Chart ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Actual Price", color="blue")
    ax.scatter(future_date, predicted_price, color="purple", s=100, label="Predicted Price")
    ax.set_title(f"Actual vs Predicted Price: {stock_choice}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (OMR)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ---- Compare Omantel vs Ooredoo ----
    df_omantel_full = process_stock_file(files_dict["Omantel.xlsx"])
    df_ooredoo_full = process_stock_file(files_dict["Ooredoo.xlsx"])

    df_omantel = df_omantel_full[(df_omantel_full["Date"].dt.date >= start_date) & (df_omantel_full["Date"].dt.date <= end_date)].copy()
    df_ooredoo = df_ooredoo_full[(df_ooredoo_full["Date"].dt.date >= start_date) & (df_ooredoo_full["Date"].dt.date <= end_date)].copy()
    df_omantel = df_omantel.sort_values("Date").dropna()
    df_ooredoo = df_ooredoo.sort_values("Date").dropna()

    # ensure enough data for compare
    if len(df_omantel) >= 60 and len(df_ooredoo) >= 60:
        stock1, stock2 = compare_stocks("Omantel", df_omantel, "Ooredoo", df_ooredoo, horizon_days)

        st.subheader("Stock Comparison: Omantel vs Ooredoo")
        st.write(pd.DataFrame([stock1, stock2]))

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(
            ["Omantel", "Ooredoo"],
            [stock1["Profit %"], stock2["Profit %"]],
            color=["green" if stock1["Profit %"] > 0 else "red",
                   "green" if stock2["Profit %"] > 0 else "red"]
        )
        ax2.set_ylabel("Expected Profit/Loss (%)")
        ax2.set_title("Expected Profit/Loss per Stock")
        st.pyplot(fig2)
    else:
        st.info("Comparison needs a bit more data in the selected period. Try a longer period (e.g., 1 Month or 1 Year).")

    # ---- Back to Home Button ----
    if st.button("🏠 Back to Home"):
        st.session_state['start_analysis'] = False


