import streamlit as st, requests, os
API = os.getenv("API_URL", "http://localhost:8000")

st.title("Heart Disease Risk Checker")
age = st.slider("Age", 20, 90, 50)
# … (add the rest of the inputs) …
if st.button("Predict"):
    payload = { "age": age, "trestbps": trestbps, … }
    resp = requests.post(f"{API}/predict", json=payload).json()
    st.write("Prediction:", "❤️ Risk" if resp["prediction"] else "✅ Low risk")
    st.write("Probability:", resp["probability"])