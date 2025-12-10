import cv2
import numpy as np
# --- PARCHE DE SEGURIDAD (HACK) ---
# Esto evita que Ultralytics descargue la versión CPU y rompa tu configuración AMD
from ultralytics.utils import checks
print("🛑 Bloqueando descargas automáticas de Ultralytics...")
checks.check_requirements = lambda *args, **kwargs: True
# ----------------------------------

from ultralytics import YOLO
import time
import onnxruntime as ort # Importamos esto para confirmar que usa DirectML

# --- 1. CONFIGURACIÓN ---
MODEL_PATH = 'MiModelo_YOLO_BEST.onnx' 
TAMANO_MODELO = 640 

CATALOGO_PRECIOS = {
    'Aceite': 35.00, 'Atun': 18.50, 'Leche': 28.00,
    'Refresco': 15.00, 'Sopa': 12.00, 'Yogurt': 10.00
}

# --- 2. GESTOR INTERFAZ ---
class GestorInterfaz:
    def __init__(self, ancho_sidebar=320):
        self.ancho = ancho_sidebar
        self.ultimo_crop = None 
        self.ultimo_nombre = "" 
        self.mensaje_estado = "Esperando..."

    def actualizar_crop(self, frame, caja, nombre_clase):
        x1, y1, x2, y2 = caja
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            try:
                crop = frame[y1:y2, x1:x2]
                self.ultimo_crop = cv2.resize(crop, (self.ancho - 40, 200))
                self.ultimo_nombre = nombre_clase 
            except: pass

    def dibujar_sidebar(self, frame_alto, carrito, total):
        h_frame, w_frame, _ = frame_alto.shape
        sidebar = np.zeros((h_frame, self.ancho, 3), dtype=np.uint8)
        
        cv2.putText(sidebar, "ULTIMO ESCANEO:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if self.ultimo_crop is not None:
            cv2.putText(sidebar, self.ultimo_nombre.upper(), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            h_c, w_c, _ = self.ultimo_crop.shape
            if 75 + h_c < h_frame:
                sidebar[75:75+h_c, 20:20+w_c] = self.ultimo_crop
        
        y_pos = 320 
        cv2.putText(sidebar, "--- TICKET ---", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_pos += 30
        for prod, info in list(carrito.items())[-6:]: 
            cv2.putText(sidebar, f"{info['cantidad']}x {prod}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(sidebar, f"${info['subtotal']:.0f}", (self.ancho - 80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 25

        cv2.rectangle(sidebar, (0, h_frame-100), (self.ancho, h_frame), (30, 30, 30), -1)
        cv2.putText(sidebar, f"TOTAL: ${total:.2f}", (100, h_frame-20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        return np.concatenate((frame_alto, sidebar), axis=1)

# --- 3. INICIO ---
print(f"🔄 Cargando modelo GPU: {MODEL_PATH} ...")
print(f"ℹ️ Proveedores disponibles: {ort.get_available_providers()}")

# Intentamos cargar forzando providers manualmente por si acaso
try:
    model = YOLO(MODEL_PATH, task='detect')
    # Inyección manual de providers para asegurar DirectML
    if hasattr(model.predictor, 'model') and hasattr(model.predictor.model, 'sess'):
        model.predictor.model.sess.set_providers(['DmlExecutionProvider'])
except Exception as e:
    print(f"Nota: {e}")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

ui = GestorInterfaz(ancho_sidebar=350)
ids_procesados = set()
carrito = {}
total = 0.0

print("✅ SISTEMA LISTO (AMD/DirectML Protegido).")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Tracking
    results = model.track(frame, persist=True, verbose=False, imgsz=TAMANO_MODELO, tracker="bytetrack.yaml")

    for r in results:
        boxes = r.boxes
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            clases = boxes.cls.cpu().numpy().astype(int)
            nombres = r.names
            cajas = boxes.xyxy.cpu().numpy().astype(int)

            for id_obj, cls, box in zip(ids, clases, cajas):
                nombre = nombres[cls]
                precio = CATALOGO_PRECIOS.get(nombre, 0)
                
                # Dibujo
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                lbl = f"{nombre.upper()} ${precio}"
                (w_t, h_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w_t + 10, y1), (0, 0, 0), -1)
                cv2.putText(frame, lbl, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Cobro
                if id_obj not in ids_procesados:
                    ids_procesados.add(id_obj)
                    if nombre not in carrito: carrito[nombre] = {'cantidad': 0, 'subtotal': 0}
                    carrito[nombre]['cantidad'] += 1
                    carrito[nombre]['subtotal'] += precio
                    total += precio
                    ui.actualizar_crop(frame, box, nombre)
                    print(f"💰 Cobrado: {nombre}")

    try:
        vis_final = ui.dibujar_sidebar(frame, carrito, total)
        cv2.imshow("Tiendita IA - GPU AMD", cv2.resize(vis_final, (1300, 650)))
    except:
        cv2.imshow("Tiendita IA - GPU AMD", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('t'): 
        ids_procesados.clear()
        carrito.clear()
        total = 0.0
        ui.ultimo_crop = None

cap.release()
cv2.destroyAllWindows()