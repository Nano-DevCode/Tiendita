"""
Compositor Responsivo de la Interfaz de Usuario en Alta Definición.
Adapta automáticamente la barra lateral y los elementos gráficos a cualquier
resolución de pantalla y cámara web (480p, 720p, 1080p, etc.).
"""

import time
import cv2
import numpy as np
import supervision as sv
from .renderer import UIRenderer
from .theme import Theme


class ResponsiveLayout:
    """Orquestador responsivo del layout visual del sistema POS."""

    @staticmethod
    def calcular_dimensiones(orig_w: int, orig_h: int):
        """
        Calcula las dimensiones ideales de video, sidebar y ventana para
        que la aplicación se abra con aspecto espacioso, proporcionado y nítido.
        """
        try:
            import ctypes
            from ctypes import wintypes
            class RECT(ctypes.Structure):
                _fields_ = [('left', wintypes.LONG), ('top', wintypes.LONG),
                            ('right', wintypes.LONG), ('bottom', wintypes.LONG)]
            rect = RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            screen_w = rect.right - rect.left
            screen_h = rect.bottom - rect.top
        except Exception:
            screen_w, screen_h = 1920, 1032

        aspect_cam = max(1.0, orig_w / max(1, orig_h))

        # Altura vertical ideal para que el sidebar tenga espacio suficiente (760 - 820 px)
        max_h_posible = int(screen_h * 0.82)
        target_h = max(720, min(max_h_posible, 800))

        # Ancho del sidebar proporcional para tarjetas amplias
        sidebar_w = int(max(440, min(500, target_h * 0.58)))

        # Ancho del video manteniendo la relación de aspecto original de la cámara (sin distorsión)
        video_w = int(target_h * aspect_cam)

        total_w = video_w + sidebar_w

        # Ajuste si supera el ancho utilizable del monitor
        if total_w > (screen_w - 60):
            factor = (screen_w - 60) / total_w
            target_h = int(target_h * factor)
            video_w = int(target_h * aspect_cam)
            sidebar_w = int(max(420, (screen_w - 60) - video_w))
            total_w = video_w + sidebar_w

        pos_x = max(0, (screen_w - total_w) // 2)
        pos_y = max(0, (screen_h - target_h) // 2 - 15)

        return total_w, target_h, video_w, sidebar_w, pos_x, pos_y

    @staticmethod
    def render(frame: np.ndarray, detections: sv.Detections, labels: list,
               corner_annotator: sv.BoxCornerAnnotator, label_annotator: sv.LabelAnnotator,
               estado, fps: float, device_name: str, clases_presentes: set,
               video_w: int, target_h: int, sidebar_w: int) -> np.ndarray:
        """
        Construye la composición final en Alta Definición:
        1. Escala el video de la cámara con interpolación cúbica de alta calidad.
        2. Dibuja anotaciones tecnológicas de alta resolución.
        3. Dibuja el HUD superior.
        4. Renderiza el panel lateral con Pillow (Segoe UI TrueType anti-aliased).
        5. Fusiona ambos componentes en un canvas sin pixelación ni deformación.
        """
        orig_h, orig_w = frame.shape[:2]

        # 1. Escalar fotograma de cámara con interpolación cúbica (alta definición)
        if (orig_w, orig_h) != (video_w, target_h):
            video_frame = cv2.resize(frame, (video_w, target_h), interpolation=cv2.INTER_CUBIC)
            scale_x = video_w / orig_w
            scale_y = target_h / orig_h

            # Escalar detecciones para renderizado nítido a resolución completa
            if detections.xyxy is not None and len(detections.xyxy) > 0:
                xyxy_scaled = detections.xyxy * np.array([scale_x, scale_y, scale_x, scale_y])
                detections_scaled = sv.Detections(
                    xyxy=xyxy_scaled,
                    confidence=detections.confidence,
                    class_id=detections.class_id,
                    tracker_id=detections.tracker_id
                )
            else:
                detections_scaled = detections
        else:
            video_frame = frame.copy()
            detections_scaled = detections

        # 2. Renderizado de cajas y etiquetas a resolución completa
        video_frame = corner_annotator.annotate(scene=video_frame, detections=detections_scaled)
        video_frame = label_annotator.annotate(scene=video_frame, detections=detections_scaled, labels=labels)

        # Factor de escala visual
        scale_factor = max(0.85, min(1.25, target_h / 760.0))

        # 3. Dibujar HUD de telemetría superior sobre el video
        tiempo_restante_toast = estado.toast_expira - time.time()
        alfa_toast = max(0.0, min(1.0, tiempo_restante_toast / 0.8))

        UIRenderer.render_hud_superior(
            frame=video_frame,
            fps=fps,
            device_name=device_name,
            modo_nombre=estado.modo_actual,
            toast_mensaje=estado.toast_mensaje if tiempo_restante_toast > 0 else None,
            toast_alfa=alfa_toast,
            descuento_activo=(estado.descuento_pct > 0),
            pausado=estado.pausado,
            scale=scale_factor
        )

        # 4. Renderizar barra lateral con tipografía Segoe UI TrueType anti-aliased (Pillow)
        sidebar = UIRenderer.render_sidebar_hd(
            width=sidebar_w,
            height=target_h,
            estado=estado,
            scale=scale_factor,
            clases_presentes=clases_presentes,
            video_offset=video_w
        )

        # 5. Fusión perfecta a resolución de destino (Exact 1:1)
        vista_completa = np.concatenate((video_frame, sidebar), axis=1)

        # Modal de borrado manual si está activo
        if estado.mostrar_panel_borrado:
            UIRenderer.render_panel_borrado(vista_completa, estado, scale=scale_factor)

        # Modal de confirmación de cobro exitoso
        elif estado.mostrar_modal_cobro:
            UIRenderer.render_modal_cobro_exitoso(vista_completa, estado, scale=scale_factor)

        # Modal de ayuda / arquitectura si está activo
        elif estado.mostrar_ayuda:
            UIRenderer.render_modal_ayuda(vista_completa, scale=scale_factor)

        return vista_completa
