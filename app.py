import streamlit as st

st.set_page_config(page_title="ARAS - Welcome", layout="wide")

# ---- Title ----
st.markdown("<h1 style='text-align: center; color: #8B307F;'>💼 Welcome to ARAS</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #6882BB;'>AI-powered Oman Stock Market Analysis</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---- Advertisement-like Boxes ----
ads = [
    ("📈 Predict stock prices before the market moves!", "#8B307F"),
    ("🤖 AI-driven confidence scores for Omantel & Ooredoo!", "#00AA00"),
    ("📊 Compare top stocks in seconds!", "#FF6600"),
    ("💡 Make smarter investment decisions today!", "#8B307F"),
    ("🚀 Track market trends like a pro!", "#00AA00")
]

for text, color in ads:
    st.markdown(f"""
    <div style='background-color:{color};padding:15px;border-radius:10px;margin-bottom:10px;color:#FFFFFF;text-align:center;font-size:18px;'>
    {text}
    </div>
    """, unsafe_allow_html=True)

# ---- Call to Action ----
st.markdown("---")
if st.button("🚀 Start Analysis"):
    st.experimental_set_query_params(page="analysis")  # سينتقل للصفحة الثانية
