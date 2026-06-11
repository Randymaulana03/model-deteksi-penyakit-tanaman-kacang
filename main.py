import os

import numpy as np
import streamlit as st
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except ModuleNotFoundError:
    from keras.models import load_model

MODEL_PATH = 'model_penyakit_daun_kacang_fixed.h5'


@st.cache_resource(show_spinner=False)
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'Model tidak ditemukan pada path: {MODEL_PATH}.\n'
            'Pastikan file model ada di folder proyek atau update path model.'
        )
    return load_model(MODEL_PATH)


def preprocess_image(image_file, target_size=(150, 150)):
    """Buka gambar, resize ke 150x150, konversi ke array, normalisasi, dan tambahkan batch dimension."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(image_file):
    model = load_model_cached()
    image = preprocess_image(image_file)
    prediction = model.predict(image, verbose=0)
    raw_score = float(prediction[0][0])

    if raw_score < 0.5:
        label = 'Sehat'
        confidence = (1 - raw_score) * 100
    else:
        label = 'Tidak Sehat'
        confidence = raw_score * 100

    return label, round(confidence, 2), round(raw_score, 4)


st.set_page_config(page_title='Deteksi Penyakit Daun Kacang', layout='centered')
st.title('Deteksi Penyakit Daun Kacang')
st.markdown(
    'Unggah gambar daun kacang (JPG/PNG) untuk mendeteksi apakah daun sehat atau terkena penyakit.'
)

uploaded_file = st.file_uploader('Pilih gambar daun kacang', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Pratinjau Gambar', use_column_width=True)
    except Exception as err:
        st.error('Gagal menampilkan gambar: ' + str(err))
        st.stop()

    if st.button('Jalankan Analisis'):
        with st.spinner('Memproses gambar...'):
            try:
                uploaded_file.seek(0)
                label, confidence, raw_score = predict(uploaded_file)
                if label == 'Sehat':
                    st.success(f'Kondisi: {label}')
                else:
                    st.error(f'Kondisi: {label}')

                st.write(f'**Tingkat keyakinan:** {confidence}%')
                st.progress(min(int(confidence), 100))
                st.write(f'**Raw score model:** {raw_score}')
            except FileNotFoundError as err:
                st.error(str(err))
            except Exception as err:
                st.error('Terjadi kesalahan saat prediksi: ' + str(err))

else:
    st.info('Silakan unggah gambar dahulu untuk memulai analisis.')
