from ultralytics import YOLO

# 1. Cargar el modelo que bajaste de Drive (.pt)
print("Cargando modelo .pt...")
model = YOLO('MiModelo_YOLO_BEST.pt')

# 2. Exportar a ONNX (Formato Universal)
print("Convirtiendo a ONNX... espera unos segundos...")
model.export(format='onnx', opset=12)

print("✅ ¡Listo! Ya tienes el archivo MiModelo_YOLO_BEST.onnx")