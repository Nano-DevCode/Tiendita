import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv # Librería profesional para tracking

# --- CONFIGURACIÓN ---
MODEL_PATH = 'MiModelo_YOLO_BEST.onnx' # Tu archivo ONNX
TAMANO_MODELO = 640  # Debe coincidir con el export
CONF_UMBRAL = 0.5    # Confianza mínima
IOU_UMBRAL = 0.45    # Para limpiar cajas duplicadas (NMS)

CATALOGO_PRECIOS = {
    'Aceite': 35.00, 'Atun': 18.50, 'Leche': 28.00,
    'Refresco': 15.00, 'Sopa': 12.00, 'Yogurt': 10.00
}

# Orden de clases (0, 1, 2...)
CLASES_NOMBRES = ['Aceite', 'Atun', 'Leche', 'Refresco', 'Sopa', 'Yogurt']

# --- MOTOR DE IA: ONNX PURO (SIN ULTRALYTICS) ---
class MotorONNX:
    def __init__(self, model_path):
        print("⚙️ Inicializando Motor DirectML (GPU)...")
        
        # CONFIGURACIÓN AGRESIVA DE GPU
        # Forzamos DmlExecutionProvider (DirectML para AMD/Intel/Nvidia en Windows)
        providers = [
            ('DmlExecutionProvider', {
                'device_id': 0, # Probamos ID 0 (Suele ser la dedicada en configuraciones primarias)
            }),
            ('DmlExecutionProvider', {
                'device_id': 1, # Probamos ID 1 por si acaso
            }),
            'CPUExecutionProvider' # Fallback solo si explota la GPU
        ]
        
        try:
            # Creamos la sesión directa
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Verificamos quién ganó
            prov = self.session.get_providers()[0]
            print(f"✅ MOTOR CARGADO EN: {prov}")
            if 'Dml' in prov:
                print("🔥 MODO ALTO RENDIMIENTO ACTIVO")
            else:
                print("⚠️ AVISO: Se está usando CPU. Revisa 'pip install onnxruntime-directml'")
                
        except Exception as e:
            print(f"❌ Error crítico cargando modelo: {e}")
            exit()

        # Obtener nombres de entrada/salida del modelo
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]

    def preprocesar(self, frame):
        # Preparar imagen para la GPU (Resize -> Normalizar -> Transponer)
        img = cv2.resize(frame, (TAMANO_MODELO, TAMANO_MODELO))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)) 
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0
        return img

    def inferir(self, frame):
        # 1. Preprocesar
        input_tensor = self.preprocesar(frame)
        
        # 2. INFERENCIA PURA (Aquí es donde la GPU trabaja al 100%)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 3. Post-procesamiento Manual (Decodificar salida YOLOv8)
        # La salida es [1, 84, 8400] -> [8400, 84]
        output = outputs[0][0].transpose()
        
        # Extraer cajas y puntuaciones
        boxes = output[:, :4] 
        scores = output[:, 4:].max(axis=1) 
        class_ids = output[:, 4:].argmax(axis=1) 
        
        # Filtrado rápido por confianza
        mask = scores > CONF_UMBRAL
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0: return sv.Detections.empty()

        # Escalar coordenadas al tamaño real del video
        h_orig, w_orig, _ = frame.shape
        scale_x = w_orig / TAMANO_MODELO
        scale_y = h_orig / TAMANO_MODELO
        
        # Convertir CX,CY,W,H -> X1,Y1,X2,Y2 y aplicar NMS
        boxes_xyxy = []
        boxes_para_nms = []
        
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            # Para NMS necesitamos (x, y, w, h) int
            x_nms = int((cx - w/2) * scale_x)
            y_nms = int((cy - h/2) * scale_y)
            w_nms = int(w * scale_x)
            h_nms = int(h * scale_y)
            boxes_para_nms.append([x_nms, y_nms, w_nms, h_nms])

        # Non-Maximum Suppression (Eliminar cajas encimadas)
        indices = cv2.dnn.NMSBoxes(boxes_para_nms, scores, CONF_UMBRAL, IOU_UMBRAL)
        
        final_xyxy = []
        final_scores = []
        final_classes = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes_para_nms[i]
                final_xyxy.append([x, y, x+w, y+h]) # Guardar como X1,Y1,X2,Y2
                final_scores.append(scores[i])
                final_classes.append(class_ids[i])

        if len(final_xyxy) == 0: return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(final_xyxy),
            confidence=np.array(final_scores),
            class_id=np.array(final_classes)
        )

# --- INTERFAZ LIMPIA (SOLO TEXTO) ---
class GestorInterfaz:
    def __init__(self, ancho=350):
        self.ancho = ancho

    def dibujar(self, frame, carrito, total):
        h, w, _ = frame.shape
        sidebar = np.zeros((h, self.ancho, 3), dtype=np.uint8)
        
        # Header
        cv2.putText(sidebar, "TIENDITA GPU", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(sidebar, (20, 60), (self.ancho-20, 60), (255, 255, 255), 1)
        
        # Lista
        y = 100
        for p, info in list(carrito.items())[-15:]:
            txt_izq = f"{info['cantidad']}x {p}"
            txt_der = f"${info['subtotal']:.0f}"
            cv2.putText(sidebar, txt_izq, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(sidebar, txt_der, (self.ancho-80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y += 30
            
        # Total
        cv2.rectangle(sidebar, (0, h-100), (self.ancho, h), (30,30,30), -1)
        cv2.putText(sidebar, "TOTAL:", (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(sidebar, f"${total:.2f}", (80, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        return np.concatenate((frame, sidebar), axis=1)

# --- PROGRAMA PRINCIPAL ---
print("🚀 Iniciando Sistema ONNX Nativo...")
motor = MotorONNX(MODEL_PATH) # Carga el motor
tracker = sv.ByteTrack()      # Carga el tracker
ui = GestorInterfaz(350)      # Carga la UI

# Variables
ids_procesados = set()
carrito = {}
total = 0.0

# Cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("✅ SISTEMA CORRIENDO. (Presiona Q para salir, T para reiniciar)")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. DETECCIÓN (GPU)
    detections = motor.inferir(frame)

    # 2. TRACKING (CPU - Supervision Library)
    detections = tracker.update_with_detections(detections)

    # 3. LÓGICA DE NEGOCIO
    labels = []
    
    if detections.tracker_id is not None:
        for xyxy, class_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
            if class_id < len(CLASES_NOMBRES):
                nombre = CLASES_NOMBRES[class_id]
                precio = CATALOGO_PRECIOS.get(nombre, 0)
                
                # Etiqueta visual para el video
                labels.append(f"#{tracker_id} {nombre}")
                
                # Lógica de cobro
                if tracker_id not in ids_procesados:
                    ids_procesados.add(tracker_id)
                    if nombre not in carrito: carrito[nombre] = {'cantidad': 0, 'subtotal': 0}
                    carrito[nombre]['cantidad'] += 1
                    carrito[nombre]['subtotal'] += precio
                    total += precio
                    print(f"💰 Cobrado: {nombre}")

    # 4. DIBUJAR EN PANTALLA
    # Usamos supervision para dibujar cajas (es muy eficiente)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    try:
        vis_final = ui.dibujar(frame, carrito, total)
        cv2.imshow("Tiendita Full GPU", cv2.resize(vis_final, (1300, 650)))
    except:
        cv2.imshow("Tiendita Full GPU", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('t'): 
        ids_procesados.clear()
        carrito.clear()
        total = 0.0
        print("🔄 Ticket reiniciado.")

cap.release()
cv2.destroyAllWindows()