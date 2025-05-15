# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle

# Load model & encoder
model = pickle.load(open('xgboost_model.pkl', 'rb'))
le_country = pickle.load(open('label_encoder_country.pkl', 'rb'))
le_category = pickle.load(open('label_encoder_category.pkl', 'rb'))

st.title("Prediksi dengan XGBoost")

# Input user
country = st.selectbox("Pilih negara:", le_country.classes_)
category = st.selectbox("Pilih kategori:", le_category.classes_)
feature1 = st.number_input("Fitur numerik 1")
feature2 = st.number_input("Fitur numerik 2")

if st.button("Prediksi"):
    # Transform input
    encoded_country = le_country.transform([country])[0]
    encoded_category = le_category.transform([category])[0]
    
    # Buat DataFrame untuk prediksi
    input_data = pd.DataFrame([[encoded_country, encoded_category, feature1, feature2]])
    
    # Prediksi
    prediction = model.predict(input_data)
    st.success(f"Hasil prediksi: {prediction[0]}")
