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
import onnxruntime as ort 

# --- 1. CONFIGURACIÓN ---
MODEL_PATH = 'MiModelo_YOLO_BEST.onnx' 
TAMANO_MODELO = 640 

CATALOGO_PRECIOS = {
    'Aceite': 35.00, 'Atun': 18.50, 'Leche': 28.00,
    'Refresco': 15.00, 'Sopa': 12.00, 'Yogurt': 10.00
}

# --- 2. GESTOR INTERFAZ (LIMPIO - SIN FOTOS) ---
class GestorInterfaz:
    def __init__(self, ancho_sidebar=320):
        self.ancho = ancho_sidebar
        # Ya no necesitamos variables para guardar fotos

    def dibujar_sidebar(self, frame_alto, carrito, total):
        h_frame, w_frame, _ = frame_alto.shape
        sidebar = np.zeros((h_frame, self.ancho, 3), dtype=np.uint8)
        
        # TÍTULO PRINCIPAL
        cv2.putText(sidebar, "TIENDITA IA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(sidebar, (20, 60), (self.ancho-20, 60), (255, 255, 255), 1)
        
        # HEADER LISTA
        y_pos = 100 
        cv2.putText(sidebar, "--- TICKET ---", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_pos += 30

        # LISTA DE COMPRAS
        # Como quitamos la foto, ahora mostramos los últimos 15 productos (antes eran 6)
        for prod, info in list(carrito.items())[-15:]: 
            texto_izq = f"{info['cantidad']}x {prod}"
            texto_der = f"${info['subtotal']:.0f}"
            
            cv2.putText(sidebar, texto_izq, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(sidebar, texto_der, (self.ancho - 80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 25

        # TOTAL (Siempre fijo abajo)
        cv2.rectangle(sidebar, (0, h_frame-100), (self.ancho, h_frame), (30, 30, 30), -1)
        cv2.putText(sidebar, "TOTAL A PAGAR:", (20, h_frame-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(sidebar, f"${total:.2f}", (80, h_frame-20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        
        return np.concatenate((frame_alto, sidebar), axis=1)

# --- 3. INICIO ---
print(f"🔄 Cargando modelo GPU: {MODEL_PATH} ...")
print(f"ℹ️ Proveedores disponibles: {ort.get_available_providers()}")

try:
    model = YOLO(MODEL_PATH, task='detect')
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

print("✅ SISTEMA LISTO (AMD/DirectML Protegido - SOLO TEXTO).")

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
                
                # Dibujo en el video (Cuadro y Precio)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                lbl = f"{nombre.upper()} ${precio}"
                (w_t, h_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w_t + 10, y1), (0, 0, 0), -1)
                cv2.putText(frame, lbl, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Cobro (Lógica)
                if id_obj not in ids_procesados:
                    ids_procesados.add(id_obj)
                    if nombre not in carrito: carrito[nombre] = {'cantidad': 0, 'subtotal': 0}
                    carrito[nombre]['cantidad'] += 1
                    carrito[nombre]['subtotal'] += precio
                    total += precio
                    # YA NO LLAMAMOS A ACTUALIZAR_CROP
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

cap.release()
cv2.destroyAllWindows()