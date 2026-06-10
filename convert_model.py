"""
Script untuk convert model .h5 lama (TensorFlow 1.x) ke format TensorFlow 2.13.0
"""
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

MODEL_OLD = 'model_penyakit_daun_kacang.h5'
MODEL_NEW = 'model_penyakit_daun_kacang_converted.h5'

def convert_model():
    print(f"Mencoba load model lama: {MODEL_OLD}")
    
    if not os.path.exists(MODEL_OLD):
        print(f"❌ File {MODEL_OLD} tidak ditemukan!")
        return False
    
    try:
        # Coba load dengan berbagai cara
        print("\n[Percobaan 1] Load dengan load_model biasa...")
        model = load_model(MODEL_OLD)
        print("✓ Berhasil!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\n[Percobaan 2] Load dengan safe_mode=False...")
        try:
            model = load_model(MODEL_OLD, safe_mode=False)
            print("✓ Berhasil!")
        except:
            print("✗ Gagal juga.")
            print("\n[Percobaan 3] Rebuild model dari config...")
            try:
                with open(MODEL_OLD.replace('.h5', '_config.json'), 'r') as f:
                    config = json.load(f)
                model = tf.keras.models.model_from_json(config)
                print("✓ Berhasil!")
            except:
                print("✗ Semua percobaan gagal. Model tidak bisa di-load.")
                return False
    
    # Simpan model baru
    print(f"\nMenyimpan model yang sudah dikonversi ke: {MODEL_NEW}")
    try:
        model.save(MODEL_NEW, save_format='h5')
        print(f"✓ Model berhasil disimpan!")
        
        # Coba load model baru untuk verify
        print(f"\nMemverifikasi model baru...")
        test_model = load_model(MODEL_NEW)
        print(f"✓ Model berhasil diload kembali!")
        print(f"\nModel Architecture:")
        test_model.summary()
        
        print(f"\n" + "="*60)
        print(f"✓ Konversi BERHASIL!")
        print(f"="*60)
        print(f"Gunakan model baru: {MODEL_NEW}")
        return True
        
    except Exception as e:
        print(f"✗ Error saat menyimpan: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("Model Converter - TensorFlow 1.x ke 2.13.0")
    print("="*60)
    convert_model()
