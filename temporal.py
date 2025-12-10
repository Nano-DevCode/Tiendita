import onnxruntime as ort
try:
    print(f"Versión instalada: {ort.__version__}")
    print(f"Proveedores: {ort.get_available_providers()}")
    print("✅ ¡TODO CORRECTO!")
except AttributeError:
    print("❌ Sigue fallando. Revisa si tienes un archivo 'onnxruntime.py' en tu carpeta.")