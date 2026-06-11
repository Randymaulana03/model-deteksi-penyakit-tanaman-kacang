import os

import numpy as np
import streamlit as st
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except ModuleNotFoundError:
    from keras.models import load_model

MODEL_PATH = 'model_penyakit_daun_kacang_fixed.h5'


# Custom CSS untuk tema hijau
st.markdown("""
<style>
    :root {
        --primary-green: #2E7D32;
        --light-green: #4CAF50;
        --lighter-green: #81C784;
        --pale-green: #C8E6C9;
        --dark-green: #1B5E20;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        padding: 40px 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(46, 125, 50, 0.2);
    }
    
    .header-container h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .header-container p {
        font-size: 1.1em;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    /* Main container */
    .main-container {
        background: linear-gradient(to bottom, #f1f8f6 0%, #e8f5e9 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #2E7D32;
        margin-bottom: 20px;
    }
    
    /* Upload area styling */
    .upload-area {
        background: white;
        padding: 30px;
        border-radius: 10px;
        border: 2px dashed #4CAF50;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #2E7D32;
        background: #f9fff9;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 1.1em;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 8px rgba(46, 125, 50, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #1B5E20 0%, #2E7D32 100%);
        box-shadow: 0 6px 12px rgba(46, 125, 50, 0.3);
        transform: translateY(-2px);
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8F6 100%);
        border-left: 5px solid #2E7D32;
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.1);
    }
    
    .result-card h3 {
        color: #2E7D32;
        font-size: 1.3em;
        margin-top: 0;
    }
    
    /* Healthy status */
    .status-sehat {
        background: linear-gradient(90deg, #4CAF50 0%, #81C784 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        font-size: 1.2em;
        font-weight: 600;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    
    /* Unhealthy status */
    .status-tidaksehat {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        font-size: 1.2em;
        font-weight: 600;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.3);
    }
    
    /* Confidence bar */
    .confidence-section {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        border-top: 4px solid #4CAF50;
    }
    
    .confidence-label {
        color: #2E7D32;
        font-weight: 600;
        font-size: 1.05em;
        margin-bottom: 10px;
    }
    
    /* Info box */
    .info-box {
        background: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        color: #1B5E20;
        margin: 10px 0;
        font-size: 0.95em;
    }
    
    /* Image preview */
    .image-preview-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.1);
        border: 1px solid #C8E6C9;
    }
    
    .image-caption {
        color: #2E7D32;
        font-weight: 600;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


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


# Page config
st.set_page_config(
    page_title='Deteksi Penyakit Daun Kacang',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# Header
st.markdown("""
<div class="header-container">
    <h1>🌿 Deteksi Penyakit Daun Kacang</h1>
    <p>Sistem AI untuk Diagnosis Kesehatan Tanaman Kacang</p>
</div>
""", unsafe_allow_html=True)

# Main content
st.markdown("""
<div class="main-container">
    <p style="font-size: 1.05em; color: #2E7D32; margin: 0;">
        📸 Unggah foto daun kacang Anda untuk analisis kesehatan secara otomatis. 
        Sistem kami akan mendeteksi apakah daun dalam kondisi sehat atau terkena penyakit.
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader
st.markdown("""<div class="upload-area">""", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    'Pilih gambar daun kacang (JPG/PNG)',
    type=['jpg', 'jpeg', 'png'],
    help='Format: JPG atau PNG, ukuran maksimal 200MB'
)
st.markdown("""</div>""", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.markdown("""<div class="image-preview-container">
            <div class="image-caption">📷 Pratinjau Gambar</div>
        """, unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)
    except Exception as err:
        st.error(f'⚠️ Gagal menampilkan gambar: {str(err)}')
        st.stop()

    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button('🔍 Jalankan Analisis', use_container_width=True)

    if analyze_button:
        with st.spinner('⏳ Memproses gambar... Mohon tunggu'):
            try:
                uploaded_file.seek(0)
                label, confidence, raw_score = predict(uploaded_file)
                
                # Result card
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<h3>📊 Hasil Analisis</h3>', unsafe_allow_html=True)
                
                # Status display
                if label == 'Sehat':
                    st.markdown(f'<div class="status-sehat">✅ Daun {label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-tidaksehat">⚠️ Daun {label}</div>', unsafe_allow_html=True)
                
                # Confidence section
                st.markdown(f"""
                <div class="confidence-section">
                    <div class="confidence-label">📈 Tingkat Keyakinan: {confidence}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar
                st.progress(min(int(confidence), 100) / 100)
                
                # Additional info
                st.markdown(f"""
                <div class="info-box">
                    <strong>Score Model:</strong> {raw_score} 
                    {'(Lebih dekat ke 0 = Sehat)' if raw_score < 0.5 else '(Lebih dekat ke 1 = Tidak Sehat)'}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except FileNotFoundError as err:
                st.error(f'❌ {str(err)}')
            except Exception as err:
                st.error(f'❌ Terjadi kesalahan saat prediksi: {str(err)}')

else:
    st.markdown("""
    <div class="info-box" style="background: #C8E6C9; border-left: 4px solid #2E7D32; padding: 20px; border-radius: 8px;">
        <h3 style="color: #1B5E20; margin-top: 0;">👋 Selamat Datang</h3>
        <p style="color: #2E7D32; margin: 10px 0;">
            Silakan unggah gambar daun kacang untuk memulai analisis. Sistem akan secara otomatis:
        </p>
        <ul style="color: #2E7D32;">
            <li>✓ Menganalisis kesehatan daun</li>
            <li>✓ Mendeteksi penyakit (jika ada)</li>
            <li>✓ Memberikan tingkat akurasi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

