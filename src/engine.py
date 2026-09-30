"""
Motor de Inferencia ONNX Acelerado por Hardware.
Implementa aceleración nativa DirectML (GPU) con fallback a CPU y un pipeline
de preprocesamiento y postprocesamiento 100% vectorizado con NumPy.
"""

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from .config import (
    MODELO_PATH, TAMANO_MODELO, CONF_UMBRAL, IOU_UMBRAL,
    MIN_AREA_PCT, MAX_AREA_PCT, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO
)


class MotorONNX:
    """
    Motor de inferencia de visión artificial sobre ONNX Runtime.
    Soporta GPUs AMD Radeon, Nvidia GeForce/RTX e Intel Iris mediante DirectML.
    """
    def __init__(self, model_path: str = MODELO_PATH, tamano: int = TAMANO_MODELO,
                 forzar_cpu: bool = False):
        self.tamano = tamano
        self.conf_umbral = CONF_UMBRAL
        self.iou_umbral = IOU_UMBRAL

        # Opciones avanzadas del runtime
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if forzar_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = [
                ('DmlExecutionProvider', {'device_id': 0}),
                'CPUExecutionProvider'
            ]

        try:
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            self.active_provider = self.session.get_providers()[0]
            if 'Dml' in self.active_provider:
                self.device_name = "GPU DirectML (Acelerado)"
            else:
                self.device_name = "CPU (Optimizado AVX2)"
            print(f"[OK] Motor cargado en: {self.active_provider} [{self.device_name}]")
        except Exception as e:
            print(f"[ERROR] Error al iniciar sesion de inferencia ONNX: {e}")
            raise SystemExit(e)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    @staticmethod
    def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        """
        Redimensiona preservando la relación de aspecto original de la imagen
        y rellenando los bordes con padding neutro (formato estándar de YOLO).
        """
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2.0
        dh /= 2.0

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def preprocesar(self, frame: np.ndarray):
        """
        Prepara la imagen de entrada con letterboxing estricto (sin distorsión vertical),
        conversión de color BGR -> RGB y normalización a rango [0.0, 1.0].
        """
        img_lb, r, (dw, dh) = self.letterbox(frame, (self.tamano, self.tamano))
        img = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)), dtype=np.float32)
        img *= (1.0 / 255.0)
        return np.expand_dims(img, axis=0), r, (dw, dh)

    def inferir(self, frame: np.ndarray) -> sv.Detections:
        """
        Ejecuta la inferencia sobre el tensor y procesa la salida decodificando
        cajas, probabilidades de clase y aplicando NMS de forma vectorizada.
        """
        tensor, r, (dw, dh) = self.preprocesar(frame)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        output = outputs[0][0].T  # Shape transpuesta: [8400, 10]

        boxes = output[:, :4]
        probs = output[:, 4:]

        scores = np.max(probs, axis=1)
        class_ids = np.argmax(probs, axis=1)

        mask = scores >= self.conf_umbral
        if not np.any(mask):
            return sv.Detections.empty()

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Deshacer el padding y escalar con la razón 'r' exacta del letterbox
        h_orig, w_orig = frame.shape[:2]
        cx = (boxes[:, 0] - dw) / r
        cy = (boxes[:, 1] - dh) / r
        bw = boxes[:, 2] / r
        bh = boxes[:, 3] / r

        # Validación geométrica estricta de productos (descarta ruidos, teclas, tiras)
        areas = bw * bh
        frame_area = w_orig * h_orig
        ratios = bw / np.maximum(bh, 1e-5)

        valid_geo = (
            (areas >= frame_area * MIN_AREA_PCT) &
            (areas <= frame_area * MAX_AREA_PCT) &
            (ratios >= MIN_ASPECT_RATIO) &
            (ratios <= MAX_ASPECT_RATIO)
        )

        if not np.any(valid_geo):
            return sv.Detections.empty()

        cx = cx[valid_geo]
        cy = cy[valid_geo]
        bw = bw[valid_geo]
        bh = bh[valid_geo]
        scores = scores[valid_geo]
        class_ids = class_ids[valid_geo]

        x1 = np.clip(cx - (bw * 0.5), 0, w_orig)
        y1 = np.clip(cy - (bh * 0.5), 0, h_orig)

        boxes_nms = np.stack([x1, y1, bw, bh], axis=1).astype(int).tolist()
        scores_list = scores.tolist()

        indices = cv2.dnn.NMSBoxes(boxes_nms, scores_list, self.conf_umbral, self.iou_umbral)
        if len(indices) == 0:
            return sv.Detections.empty()

        indices = np.array(indices).flatten()

        sel_x1 = x1[indices]
        sel_y1 = y1[indices]
        sel_x2 = np.clip(sel_x1 + bw[indices], 0, w_orig)
        sel_y2 = np.clip(sel_y1 + bh[indices], 0, h_orig)

        final_xyxy = np.stack([sel_x1, sel_y1, sel_x2, sel_y2], axis=1)
        final_scores = scores[indices]
        final_classes = class_ids[indices]

        return sv.Detections(
            xyxy=final_xyxy,
            confidence=final_scores,
            class_id=final_classes
        )
