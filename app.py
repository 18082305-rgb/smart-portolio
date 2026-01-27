import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="ARAS - Smart Portfolio", layout="wide")

# ---- Welcome Page ----
st.markdown("<h1 style='text-align: center; color: #8B307F;'>💼 Welcome to ARAS</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #6882BB;'>AI-powered Oman Stock Market Analysis</h3>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<p style='text-align:center; font-size:18px;'>
Predict stock prices, view confidence scores, and compare top stocks like Omantel & Ooredoo.
</p>
""", unsafe_allow_html=True)

# Start Analysis button
if st.button("🚀 Start Analysis"):
    st.session_state.start_analysis = True

# Initialize session state
if "start_analysis" not in st.session_state:
    st.session_state.start_analysis = False

# --------------------
# Main app (after button)
# --------------------
if st.session_state.start_analysis:

    st.success("ARAS Loaded! Stock analysis starts below...")

    # ---- Files are preloaded ----
    files_dict = {
        "Omantel.xlsx": "Omantel.xlsx",
        "Ooredoo.xlsx": "Ooredoo.xlsx"
    }

    stock_choice = st.selectbox("Select Stock", list(files_dict.keys()))

    horizon_days = st.selectbox(
        "Prediction Horizon",
        [("1 Day", 1), ("1 Week", 5), ("1 Month", 22), ("1 Year", 252)],
        format_func=lambda x: x[0]
    )[1]

    # ----- Helper functions -----
    def process_stock_file(file):
        df = pd.read_excel(file)
        df = df[df.iloc[:,0].astype(str).str.contains(r"\d", regex=True)]
        df.columns = ["Date","Open","High","Low","Close","Volume"]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("Date")
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df.dropna()

    def predict_price(df, horizon):
        features = ["Open","High","Low","Volume","MA5","MA10","RSI"]
        X = df[features]
        y = df["Close"].shift(-horizon)
        X = X.iloc[:-horizon]
        y = y.iloc[:-horizon]
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        predicted = model.predict(X.iloc[-1].values.reshape(1,-1))[0]
        return predicted, model, X, y

    def confidence_score(model, X_test, y_test):
        mae = mean_absolute_error(y_test, model.predict(X_test))
        error_conf = max(0, 1 - mae / y_test.mean())
        tree_preds = np.array([tree.predict(X_test.iloc[-1].values.reshape(1,-1))[0] for tree in model.estimators_])
        stability = 1 / (1 + np.std(tree_preds))
        confidence = (0.6 * error_conf + 0.4 * stability) * 100
        return min(95, max(40, confidence))

    def compare_stocks(name1, df1, name2, df2, horizon):
        def analyze(df, name):
            pred, model, X, y = predict_price(df, horizon)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            conf = confidence_score(model, X_test, y_test)
            last = df.iloc[-1]
            profit_pct = (pred - last["Close"]) / last["Close"] * 100
            trend = "Up ✅" if profit_pct > 0 else "Down ❌"
            return {
                "Name": name,
                "Last Close": last["Close"],
                "Predicted": pred,
                "Profit %": profit_pct,
                "Trend": trend,
                "Confidence": conf
            }
        return analyze(df1, name1), analyze(df2, name2)

    # ---- Main Analysis ----
    df = process_stock_file(files_dict[stock_choice])
    predicted_price, model, X, y = predict_price(df, horizon_days)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    confidence = confidence_score(model, X_test, y_test)
    current_price = df.iloc[-1]["Close"]
    profit_pct = (predicted_price - current_price)/current_price*100
    future_date = df.iloc[-1]["Date"] + pd.Timedelta(days=horizon_days)

    if profit_pct > 1:
        recommendation, rec_color = "Buy 📈", "#00AA00"
    elif profit_pct < -1:
        recommendation, rec_color = "Avoid/Sell 📉", "#FF0000"
    else:
        recommendation, rec_color = "Hold ⚪", "#FFA500"

    st.subheader(f"Stock Report: {stock_choice}")
    st.markdown(f"""
    - **Current Price:** {current_price:.3f} OMR  
    - **Predicted Price ({horizon_days} days):** {predicted_price:.3f} OMR  
    - **Profit Expectation:** {profit_pct:.2f}%  
    - **Confidence Score:** {confidence:.1f}%  
    - **Recommendation:** <span style="color:{rec_color}">{recommendation}</span>
    """, unsafe_allow_html=True)

    # Actual vs Predicted chart
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["Date"], df["Close"], label="Actual Price", color="blue")
    ax.scatter(future_date, predicted_price, color="purple", s=100, label="Predicted Price")
    ax.set_title(f"Actual vs Predicted Price: {stock_choice}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (OMR)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ---- Compare Omantel vs Ooredoo ----
    df_omantel = process_stock_file(files_dict["Omantel.xlsx"])
    df_ooredoo = process_stock_file(files_dict["Ooredoo.xlsx"])
    stock1, stock2 = compare_stocks("Omantel", df_omantel, "Ooredoo", df_ooredoo, horizon_days)

    st.subheader("Stock Comparison: Omantel vs Ooredoo")
    st.write(pd.DataFrame([stock1, stock2]))

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.bar(["Omantel", "Ooredoo"], [stock1["Profit %"], stock2["Profit %"]],
            color=["green" if stock1["Profit %"]>0 else "red",
                   "green" if stock2["Profit %"]>0 else "red"])
    ax2.set_ylabel("Expected Profit/Loss (%)")
    ax2.set_title("Expected Profit/Loss per Stock")
    st.pyplot(fig2)
