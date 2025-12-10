import onnxruntime as ort

print("Proveedores disponibles en tu PC:")
print(ort.get_available_providers())

# Debería salir algo como: ['DmlExecutionProvider', 'CPUExecutionProvider']
# Si 'DmlExecutionProvider' sale primero, ¡YA ESTÁ!