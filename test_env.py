import importlib

modules = {
    'tensorflow': 'TensorFlow',
    'flask': 'Flask',
    'PIL': 'Pillow',
    'numpy': 'NumPy',
}

for module_name, display_name in modules.items():
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f'{display_name} versi {version} berhasil dimuat')
    except Exception as e:
        print(f'Gagal memuat {display_name}: {e}')
