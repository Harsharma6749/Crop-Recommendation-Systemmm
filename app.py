import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("🌾 Crop Recommendation System")

# Input sliders
N = st.slider("Nitrogen (N)", 0, 140, 50)
P = st.slider("Phosphorus (P)", 5, 145, 50)
K = st.slider("Potassium (K)", 5, 205, 50)
temperature = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
ph = st.slider("pH", 3, 10, 6)
rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

# Predict button
if st.button("Recommend Crop"):
    model = joblib.load("crop_recommendation_model.pkl")
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(data)
    st.success(f"✅ Recommended Crop: {prediction[0]}")
