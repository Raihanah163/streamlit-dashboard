import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Startup", layout="centered")
st.title("🚀 Prediksi Keberhasilan Startup")

# Load model dan encoder
try:
    model = joblib.load('voting_ensemble_model.pkl')
    le_country = joblib.load('label_encoder_country.pkl')
    le_category = joblib.load('label_encoder_category.pkl')
except Exception as e:
    st.error(f"Gagal memuat model atau encoder: {e}")
    st.stop()

# Buat form input
with st.form("form_prediksi"):
    name = st.text_input("Nama Startup", value="Contoh Startup")
    funding_total_usd = st.number_input("Total Pendanaan (USD)", min_value=0.0, step=10000.0)
    funding_rounds = st.number_input("Jumlah Putaran Pendanaan", min_value=0, step=1)
    
    # Input tanggal berdiri
    founded_at = st.date_input("Tanggal Berdiri", min_value=datetime(1970, 1, 1), max_value=datetime.today())
    
    # Pilih negara dan kategori (gunakan label dari encoder jika perlu)
    country_code = st.text_input("Kode Negara (misalnya: USA, IND, GBR)", value="USA")
    category_list = st.text_input("Kategori (misalnya: Health, Finance, AI)", value="Health")

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    try:
        # Hitung umur startup
        startup_age = (pd.to_datetime('today') - pd.to_datetime(founded_at)).days // 365
        
        # Buat dataframe tunggal
        input_data = pd.DataFrame([{
            "funding_total_usd": funding_total_usd,
            "funding_rounds": funding_rounds,
            "country_code": country_code,
            "category_list": category_list,
            "startup_age": startup_age
        }])

        # Encode country_code dan category_list dengan handling label yang tidak dikenal
        try:
            input_data["country_code"] = le_country.transform(input_data["country_code"].astype(str))
        except ValueError:
            # Jika label tidak dikenali, gunakan label default 'unknown'
            input_data["country_code"] = le_country.transform(['unknown'])[0]
        
        try:
            input_data["category_list"] = le_category.transform(input_data["category_list"].astype(str))
        except ValueError:
            # Jika label tidak dikenali, gunakan label default 'unknown'
            input_data["category_list"] = le_category.transform(['unknown'])[0]

        # Prediksi
        pred = model.predict(input_data)[0]
        hasil = "✅ Berpotensi Sukses" if pred == 1 else "⚠️ Berpotensi Gagal"
        st.subheader("Hasil Prediksi:")
        st.write(f"**{name}** diprediksi: **{hasil}**")

        # Tampilkan detail input
        st.write("🔎 Detail Data:")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")
