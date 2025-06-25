import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("fraud_model.pkl")

# Page config
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

# Custom CSS for dark theme + styling
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: white;
        }
        .stButton>button {
            background-color: #00C9A7;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        .stTextInput>div>div>input {
            background-color: #20232a;
            color: white;
        }
        .prediction-success {
            color: #00C9A7;
            font-weight: bold;
            font-size: 24px;
        }
        .prediction-fail {
            color: #FF4B4B;
            font-weight: bold;
            font-size: 24px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 style='text-align: center;'>💳 AI Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("### 🚀 Enter Transaction Details Below:")

# Input fields
v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)
v4 = st.number_input("V4", value=0.0)
amount = st.number_input("Amount (₹)", value=0.0)

# Predict button
if st.button("🔍 Predict Fraud"):
    input_data = np.array([[v1, v2, v3, v4, amount]])
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1] * 100

    if prediction[0] == 1:
        st.markdown(f"<p class='prediction-fail'>❌ Fraudulent Transaction Detected! ({prediction_proba:.2f}% confidence)</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='prediction-success'>✅ Legitimate Transaction ({100 - prediction_proba:.2f}% confidence)</p>", unsafe_allow_html=True)

    # Explanation
    st.markdown("#### 📌 Model Explanation:")
    st.markdown("""
        - This model uses Random Forest Classifier.
        - The prediction is based on anonymized transaction features (V1–V4, Amount).
        - Confidence score shows how sure the model is about its prediction.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Made with ❤️ by Neha Vinod Varma | AIML Engineer</p>", unsafe_allow_html=True)
