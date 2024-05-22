import streamlit as st
import requests

st.title('Sentimen Analisis Pengguna Media Sosial Terhadap Pemilu 2024')

user_input = st.text_area("Masukkan teks media sosial:")

if st.button('Prediksi'):
    if user_input:
        response = requests.post("http://127.0.0.1:5000/predict", json={'text': user_input})
        prediction = response.json()['prediction']
        st.write(f"Sentimen: {prediction}")
    else:
        st.write("Masukkan teks terlebih dahulu!")
