# pages/stock_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.title("📊 ARAS Stock Analysis")

# ---- Files preloaded ----
files_dict = {
    "Omantel.xlsx": "Omantel.xlsx",
    "Ooredoo.xlsx": "Ooredoo.xlsx"
}

# ---- User selects stock and prediction horizon safely ----
stock_choice = st.selectbox("Select Stock", list(files_dict.keys()))
horizon_days = st.selectbox(
    "Prediction Horizon",
    [("1 Day", 1), ("1 Week", 5), ("1 Month", 22), ("1 Year", 252)],
    format_func=lambda x: x[0]
)[1]

# ----- Helper functions -----
def process_stock_file(file):
    try:
        df = pd.read_excel(file)
    except FileNotFoundError:
        st.error(f"File {file} not found! Please make sure the file exists.")
        st.stop()
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

# ---- Main Analysis ----
df = process_stock_file(files_dict[stock_choice])
predicted_price, model, X, y = predict_price(df, horizon_days)

# ---- Split data safely ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

confidence = confidence_score(model, X_test, y_test)
current_price = df.iloc[-1]["Close"]
profit_pct = (predicted_price - current_price)/current_price*100
future_date = df.iloc[-1]["Date"] + pd.Timedelta(days=horizon_days)

# ---- Recommendation color ----
if profit_pct > 1:
    recommendation, rec_color = "Buy 📈", "#00AA00"
elif profit_pct < -1:
    recommendation, rec_color = "Avoid/Sell 📉", "#FF0000"
else:
    recommendation, rec_color = "Hold ⚪", "#FFA500"

# ---- Display report ----
st.subheader(f"Stock Report: {stock_choice}")
st.markdown(f"""
- **Current Price:** {current_price:.3f} OMR  
- **Predicted Price ({horizon_days} days):** {predicted_price:.3f} OMR  
- **Profit Expectation:** {profit_pct:.2f}%  
- **Confidence Score:** {confidence:.1f}%  
- **Recommendation:** <span style="color:{rec_color}">{recommendation}</span>
""", unsafe_allow_html=True)

# ---- Chart: Actual vs Predicted ----
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df["Date"], df["Close"], label="Actual Price", color="blue")
ax.scatter(future_date, predicted_price, color="purple", s=100, label="Predicted Price")
ax.set_title(f"Actual vs Predicted Price: {stock_choice}")
ax.set_xlabel("Date")
ax.set_ylabel("Price (OMR)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
