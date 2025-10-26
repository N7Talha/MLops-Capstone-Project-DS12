import streamlit as st, requests, os
API = os.getenv("API_URL", "http://localhost:8000")

st.title("Heart Disease Risk Checker")
age = st.slider("Age", 20, 90, 50)
# collect every widget value
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest-pain type", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP (mmHg)", 80, 220, 130)
chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 250)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max heart rate achieved", 60, 220, 150)
exang = st.selectbox("Exercise-induced angina", [0, 1])
oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of peak ST", [0, 1, 2])
ca = st.number_input("Number of major vessels (0-3)", 0, 3, 0)
thal = st.selectbox("Thalassemia", [0, 1, 2])

if st.button("Predict"):
    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    resp = requests.post(f"{API}/predict", json=payload).json()
    st.write("Prediction:", "❤️ Risk" if resp["prediction"] else "✅ Low risk")
    st.write("Probability:", resp["probability"])