import os
import numpy as np
import streamlit as st
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except ModuleNotFoundError:
    from keras.models import load_model

MODEL_PATH = 'model_penyakit_daun_kacang_fixed.h5'

# --- 1. MENIRU FLASK: Penanganan Error Saat Memuat Model ---
@st.cache_resource(show_spinner=False)
def load_model_cached():
    try:
        loaded_model = load_model(MODEL_PATH)
        print(f"✓ Model berhasil dimuat dari {MODEL_PATH}")
        return loaded_model
    except Exception as e:
        print(f"✗ Gagal memuat model dari {MODEL_PATH}: {e}")
        return None

# Panggil fungsi agar model dimuat (atau menghasilkan None) saat awal dijalankan
model = load_model_cached()


def preprocess_image(image_file, target_size=(150, 150)):
    """Buka gambar, resize ke 150x150, konversi ke array, normalisasi, dan tambahkan batch dimension."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(image_file, loaded_model):
    image = preprocess_image(image_file)
    prediction = loaded_model.predict(image, verbose=0)
    
    # Asumsi output model biner
    raw_score = float(prediction[0][0])
    
    # Hitung confidence untuk setiap kelas
    if raw_score < 0.5:
        label = 'Sehat'
        confidence = (1 - raw_score) * 100
    else:
        label = 'Tidak Sehat'
        confidence = raw_score * 100
    
    return label, round(confidence, 2), round(raw_score, 4)


# --- ANTARMUKA STREAMLIT ---
st.set_page_config(page_title='Deteksi Penyakit Daun Kacang', layout='centered')
st.title('Deteksi Penyakit Daun Kacang')
st.markdown('Unggah gambar daun kacang (JPG/PNG) untuk mendeteksi apakah daun sehat atau terkena penyakit.')

uploaded_file = st.file_uploader('Pilih gambar daun kacang', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Pratinjau Gambar', use_column_width=True)
    except Exception as err:
        st.error('Gagal menampilkan gambar: ' + str(err))
        st.stop()

    if st.button('Jalankan Analisis'):
        
        # --- 2. MENIRU FLASK: Cek model sebelum memproses gambar ---
        if model is None:
            st.error('Status: Error | Pesan: Model belum dimuat. Pastikan file model_penyakit_daun_kacang_fixed.h5 ada.')
        else:
            with st.spinner('Memproses gambar...'):
                try:
                    uploaded_file.seek(0)
                    label, confidence, raw_score = predict(uploaded_file, model)
                    
                    # --- 3. MENIRU FLASK: Format Respons ---
                    if label == 'Sehat':
                        st.success(f'Status: Sukses | Hasil Prediksi: **{label}**')
                    else:
                        st.error(f'Status: Sukses | Hasil Prediksi: **{label}**')

                    st.write(f'**Confidence:** {confidence}%')
                    st.progress(min(int(confidence), 100))
                    st.write(f'**Raw Score:** {raw_score}')
                    
                except Exception as err:
                    st.error(f'Status: Error | Pesan: Gagal memproses gambar: {str(err)}')

else:
    st.info('Silakan unggah gambar dahulu untuk memulai analisis.')