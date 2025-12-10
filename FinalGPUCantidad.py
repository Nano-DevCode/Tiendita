import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

# --- CONFIGURACIÓN ---
MODEL_PATH = 'MiModelo_YOLO_BEST.onnx'
TAMANO_MODELO = 640
CONF_UMBRAL = 0.5    
IOU_UMBRAL = 0.45    

# Nombres de las clases (Tu inventario posible)
CLASES_NOMBRES = ['Aceite', 'Atun', 'Leche', 'Refresco', 'Sopa', 'Yogurt']

# --- MOTOR DE IA: ONNX PURO (GPU) ---
class MotorONNX:
    def __init__(self, model_path):
        print("⚙️ Inicializando Motor de Inventario (GPU)...")
        # Configuración para forzar GPU AMD/NVIDIA
        providers = [
            ('DmlExecutionProvider', {'device_id': 0}),
            ('DmlExecutionProvider', {'device_id': 1}),
            'CPUExecutionProvider'
        ]
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            prov = self.session.get_providers()[0]
            print(f"✅ MOTOR CARGADO EN: {prov}")
        except Exception as e:
            print(f"❌ Error crítico: {e}")
            exit()

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]

    def preprocesar(self, frame):
        img = cv2.resize(frame, (TAMANO_MODELO, TAMANO_MODELO))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)) 
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0
        return img

    def inferir(self, frame):
        input_tensor = self.preprocesar(frame)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        output = outputs[0][0].transpose()
        
        boxes = output[:, :4] 
        scores = output[:, 4:].max(axis=1) 
        class_ids = output[:, 4:].argmax(axis=1) 
        
        mask = scores > CONF_UMBRAL
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0: return sv.Detections.empty()

        h_orig, w_orig, _ = frame.shape
        scale_x = w_orig / TAMANO_MODELO
        scale_y = h_orig / TAMANO_MODELO
        
        boxes_xyxy = []
        boxes_para_nms = []
        
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            x_nms = int((cx - w/2) * scale_x)
            y_nms = int((cy - h/2) * scale_y)
            w_nms = int(w * scale_x)
            h_nms = int(h * scale_y)
            boxes_para_nms.append([x_nms, y_nms, w_nms, h_nms])

        indices = cv2.dnn.NMSBoxes(boxes_para_nms, scores, CONF_UMBRAL, IOU_UMBRAL)
        
        final_xyxy = []
        final_scores = []
        final_classes = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes_para_nms[i]
                final_xyxy.append([x, y, x+w, y+h])
                final_scores.append(scores[i])
                final_classes.append(class_ids[i])

        if len(final_xyxy) == 0: return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(final_xyxy),
            confidence=np.array(final_scores),
            class_id=np.array(final_classes)
        )

# --- INTERFAZ DE INVENTARIO ---
class GestorInterfaz:
    def __init__(self, ancho=350):
        self.ancho = ancho

    def dibujar(self, frame, inventario, total_items):
        h, w, _ = frame.shape
        sidebar = np.zeros((h, self.ancho, 3), dtype=np.uint8)
        
        # Header
        cv2.putText(sidebar, "INVENTARIO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(sidebar, (20, 60), (self.ancho-20, 60), (255, 255, 255), 1)
        
        # Tabla de Conteo
        y = 100
        cv2.putText(sidebar, "PRODUCTO", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(sidebar, "CANT", (self.ancho-60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 30

        # Dibujar cada producto y su cantidad
        for producto in CLASES_NOMBRES:
            cantidad = inventario.get(producto, 0)
            
            # Si hay stock, ponerlo en verde, si no, en gris
            color = (0, 255, 0) if cantidad > 0 else (100, 100, 100)
            
            cv2.putText(sidebar, producto, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(sidebar, str(cantidad), (self.ancho-60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Línea separadora sutil
            cv2.line(sidebar, (20, y+10), (self.ancho-20, y+10), (40, 40, 40), 1)
            y += 40
            
        # Total Global
        cv2.rectangle(sidebar, (0, h-100), (self.ancho, h), (30,30,30), -1)
        cv2.putText(sidebar, "ITEMS TOTALES:", (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(sidebar, str(total_items), (200, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        
        return np.concatenate((frame, sidebar), axis=1)

# --- PROGRAMA PRINCIPAL ---
print("🚀 Iniciando Sistema de Inventario...")
motor = MotorONNX(MODEL_PATH)
tracker = sv.ByteTrack()
ui = GestorInterfaz(350)

# Inicializar inventario en 0
inventario = {nombre: 0 for nombre in CLASES_NOMBRES}
ids_procesados = set()
total_items = 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("✅ SISTEMA LISTO.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Detectar (GPU)
    detections = motor.inferir(frame)

    # 2. Rastrear (CPU)
    detections = tracker.update_with_detections(detections)

    # 3. Lógica de Inventario
    labels = []
    
    if detections.tracker_id is not None:
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            if class_id < len(CLASES_NOMBRES):
                nombre = CLASES_NOMBRES[class_id]
                
                # Etiqueta visual: "#ID Nombre" (Sin precio)
                labels.append(f"#{tracker_id} {nombre}")
                
                # Si es un ID nuevo, aumentamos el stock
                if tracker_id not in ids_procesados:
                    ids_procesados.add(tracker_id)
                    inventario[nombre] += 1
                    total_items += 1
                    print(f"📦 Nuevo item registrado: {nombre} (Total: {inventario[nombre]})")

    # 4. Dibujar
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    try:
        vis_final = ui.dibujar(frame, inventario, total_items)
        cv2.imshow("Control de Inventario IA", cv2.resize(vis_final, (1300, 650)))
    except:
        cv2.imshow("Control de Inventario IA", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'): # 'R' para Resetear conteo
        ids_procesados.clear()
        inventario = {nombre: 0 for nombre in CLASES_NOMBRES}
        total_items = 0
        print("🔄 Inventario reiniciado.")

cap.release()
cv2.destroyAllWindows()