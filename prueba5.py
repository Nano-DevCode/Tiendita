# tiendita_yolov8_directml.py
import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque

# ----------------------- CONFIG -----------------------
MODEL_PATH = "MiModelo_YOLO_BEST.onnx"
IMG_SIZE = 640  # tamaño al que exportaste tu ONNX (YOLOv8 típico)
CONF_THRESH = 0.25
IOU_THRESH = 0.45
MAX_DETECTIONS = 300

CATALOGO_PRECIOS = {
    'Aceite': 35.00, 'Atun': 18.50, 'Leche': 28.00,
    'Refresco': 15.00, 'Sopa': 12.00, 'Yogurt': 10.00
}

# -------------------- UTILIDADES ----------------------
def letterbox(im, new_shape=(IMG_SIZE, IMG_SIZE), color=(114,114,114), stride=32):
    # Adapted to behave like YOLOv8 letterbox
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2
    dh /= 2

    # resize
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)

def xywh2xyxy(x):
    # x: [x_center, y_center, w, h] returns x1,y1,x2,y2
    x_c, y_c, w, h = x
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return [x1, y1, x2, y2]

def scale_coords(coords, r, pad, orig_shape):
    # coords in letterbox space -> original image space
    left, top = pad
    coords[:, [0,2]] -= left
    coords[:, [1,3]] -= top
    coords[:, :4] /= r
    # clip
    coords[:, 0] = coords[:, 0].clip(0, orig_shape[1]-1)
    coords[:, 1] = coords[:, 1].clip(0, orig_shape[0]-1)
    coords[:, 2] = coords[:, 2].clip(0, orig_shape[1]-1)
    coords[:, 3] = coords[:, 3].clip(0, orig_shape[0]-1)
    return coords

def box_iou(box1, box2):
    # box: x1,y1,x2,y2
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    area1 = max(0, (box1[2]-box1[0])*(box1[3]-box1[1]))
    area2 = max(0, (box2[2]-box2[0])*(box2[3]-box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def non_max_suppression(preds, conf_thresh=0.25, iou_thresh=0.45, max_det=300):
    # preds: (N, 85) -> cx,cy,w,h,conf,cls_probs...
    # returns list of detections: [x1,y1,x2,y2,conf, class_id]
    if preds is None:
        return []

    # confidence * class_conf
    xywh = preds[:, :4]
    conf = preds[:, 4:5]
    cls_conf = preds[:, 5:]
    scores = conf * cls_conf  # (N, num_classes)
    class_ids = np.argmax(scores, axis=1)
    class_scores = scores[np.arange(scores.shape[0]), class_ids]
    mask = class_scores > conf_thresh
    if not mask.any():
        return []

    xywh = xywh[mask]
    class_scores = class_scores[mask]
    class_ids = class_ids[mask]
    # convert xywh to xyxy in same scale (letterbox)
    boxes = np.array([xywh2xyxy(x) for x in xywh])
    # stack: x1,y1,x2,y2, score, class
    dets = np.concatenate((boxes, class_scores[:, None], class_ids[:, None].astype(np.float32)), axis=1)

    # sort by score desc
    order = np.argsort(-dets[:, 4])
    dets = dets[order]

    keep = []
    while dets.shape[0]:
        if len(keep) >= max_det:
            break
        a = dets[0]
        keep.append(a)
        if dets.shape[0] == 1:
            break
        rest = dets[1:]
        ious = np.array([box_iou(a[:4], r[:4]) for r in rest])
        # suppress those with iou > iou_thresh and same class
        mask_keep = ~((ious > iou_thresh) & (rest[:, 5] == a[5]))
        dets = rest[mask_keep]
    return np.array(keep) if keep else np.zeros((0,6))

# ---------------- ONNX RUNTIME (DML) -------------------
def create_session(model_path):
    print("Proveedores disponibles en ONNX Runtime:", ort.get_available_providers())
    # Forzar DML primero, fallback CPU
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    # Opcionales: sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    print("Sesión creada. Proveedores activos:", [p['name'] if isinstance(p, dict) and 'name' in p else p for p in session.get_providers()])
    return session

def run_inference(session, frame):
    # frame: original BGR image
    img, r, pad = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None, ...]  # (1,3,H,W)
    # ONNX input name:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    # Common YOLOv8 export: outputs[0] shape (1, N, 85)
    preds = outputs[0]
    if isinstance(preds, list) or isinstance(preds, tuple):
        preds = preds[0]
    preds = preds[0]  # (N,85)
    # NMS
    dets = non_max_suppression(preds, CONF_THRESH, IOU_THRESH, MAX_DETECTIONS)
    if dets.shape[0] == 0:
        return [], r, pad
    # dets columns: x1,y1,x2,y2,score,class
    dets_coords = dets[:, :4].copy()
    dets_coords = np.array(dets_coords)
    dets_full = np.concatenate((dets_coords, dets[:, 4:6]), axis=1)  # x1,y1,x2,y2,score,class
    # scale back to original image
    dets_full[:, :4] = scale_coords(dets_full[:, :4], r, pad, frame.shape)
    return dets_full, r, pad

# ---------------- SIMPLE TRACKER (IoU-based) --------------
class SimpleTracker:
    def __init__(self, max_lost=30, iou_threshold=0.3):
        # max_lost: number of frames to keep "lost" track
        self.next_id = 0
        self.tracks = {}  # id -> {box, lost}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def update(self, detections):
        # detections: array Nx6 (x1,y1,x2,y2,score,class)
        updated_tracks = {}
        used_det_idx = set()

        # match existing tracks to detections by IoU
        for tid, tk in list(self.tracks.items()):
            best_iou = 0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in used_det_idx: continue
                iou = box_iou(tk['box'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= self.iou_threshold and best_idx != -1:
                det = detections[best_idx]
                updated_tracks[tid] = {'box': det[:4].tolist(), 'lost': 0, 'class': int(det[5]), 'score': float(det[4])}
                used_det_idx.add(best_idx)
            else:
                # increment lost
                tk['lost'] += 1
                if tk['lost'] <= self.max_lost:
                    updated_tracks[tid] = tk  # keep it
                # else drop

        # create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in used_det_idx: continue
            tid = self.next_id
            self.next_id += 1
            updated_tracks[tid] = {'box': det[:4].tolist(), 'lost': 0, 'class': int(det[5]), 'score': float(det[4])}

        self.tracks = updated_tracks
        return self.tracks

# ---------------- UI / SIDEBAR (adapted from tu código) -------------
class GestorInterfaz:
    def __init__(self, ancho_sidebar=320):
        self.ancho = ancho_sidebar
        self.ultimo_crop = None 
        self.ultimo_nombre = "" 
        self.mensaje_estado = "Esperando..."

    def actualizar_crop(self, frame, caja, nombre_clase):
        x1, y1, x2, y2 = map(int, caja)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            try:
                crop = frame[y1:y2, x1:x2]
                self.ultimo_crop = cv2.resize(crop, (self.ancho - 40, 200))
                self.ultimo_nombre = nombre_clase 
            except Exception:
                pass

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
        # show last 6 items
        items = list(carrito.items())[-6:]
        for prod, info in items:
            cv2.putText(sidebar, f"{info['cantidad']}x {prod}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(sidebar, f"${info['subtotal']:.0f}", (self.ancho - 80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 25

        cv2.rectangle(sidebar, (0, h_frame-100), (self.ancho, h_frame), (30, 30, 30), -1)
        cv2.putText(sidebar, f"TOTAL: ${total:.2f}", (100, h_frame-20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        return np.concatenate((frame_alto, sidebar), axis=1)

# ---------------------- MAIN -----------------------------
def main():
    print("Creando sesión ONNX Runtime (intentando DML)...")
    session = create_session(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    ui = GestorInterfaz(ancho_sidebar=350)
    tracker = SimpleTracker(max_lost=15, iou_threshold=0.3)

    ids_procesados = set()
    carrito = {}
    total = 0.0

    print("SISTEMA LISTO. Presiona 'q' para salir, 't' para limpiar carrito.")
    fps_deque = deque(maxlen=10)
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, r, pad = run_inference(session, frame)
        # detections: Nx6 (x1,y1,x2,y2,score,class)
        if isinstance(detections, np.ndarray) and detections.shape[0] > 0:
            tracks = tracker.update(detections)
        else:
            tracks = tracker.update(np.zeros((0,6)))

        # draw tracks and charge
        for tid, tr in tracks.items():
            x1,y1,x2,y2 = map(int, tr['box'])
            cls_id = tr.get('class', 0)
            # NOTE: aquí necesitas un mapeo id->nombre; si tu ONNX preservó nombres en orden de entrenamiento
            # asumimos nombres genéricos como 'class_0', pero idealmente reemplaza por r.names
            nombre = f"class_{cls_id}"
            # Si tienes lista de nombres: NAMES = ['Aceite','Atun',...]; then nombre = NAMES[cls_id]
            # Dibujo:
            color = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            lbl = f"{nombre} ID:{tid}"
            (w_t, h_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + w_t + 10, y1), (0, 0, 0), -1)
            cv2.putText(frame, lbl, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Cobro: cuando aparece nuevo ID contado por primera vez
            if tid not in ids_procesados:
                ids_procesados.add(tid)
                # map class->nombre si posible
                nombre_producto = nombre  # reemplaza si tienes mapeo
                precio = CATALOGO_PRECIOS.get(nombre_producto, 0.0)
                if nombre_producto not in carrito:
                    carrito[nombre_producto] = {'cantidad': 0, 'subtotal': 0.0}
                carrito[nombre_producto]['cantidad'] += 1
                carrito[nombre_producto]['subtotal'] += precio
                total += precio
                ui.actualizar_crop(frame, (x1,y1,x2,y2), nombre_producto)
                print(f"💰 Cobrado: {nombre_producto} ID:{tid}")

        # dibujar UI
        try:
            vis_final = ui.dibujar_sidebar(frame, carrito, total)
            # ajustar para mostrar (opcional)
            show = cv2.resize(vis_final, (1300, 650))
            # FPS
            now = time.time()
            fps_deque.append(1.0 / (now - last_time) if now != last_time else 0)
            last_time = now
            fps = sum(fps_deque)/len(fps_deque) if fps_deque else 0
            cv2.putText(show, f"FPS: {fps:.1f}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.imshow("Tiendita IA - AMD DirectML", show)q
        except Exception as e:
            cv2.putText(frame, f"UI ERROR: {e}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            cv2.imshow("Tiendita IA - AMD DirectML", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            ids_procesados.clear()
            carrito.clear()
            total = 0.0
            ui.ultimo_crop = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
