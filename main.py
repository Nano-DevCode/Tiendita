"""
=============================================================================
TIENDITA INTELIGENTE IA - SISTEMA AUTÓNOMO DE PUNTO DE VENTA & AUDITORÍA
=============================================================================
Punto de Entrada Principal (Main Orchestrator).
Coordina el ciclo de vida de la aplicación, el motor de inferencia ONNX,
el rastreador ByteTrack, la máquina de estados y la interfaz responsiva en Alta Definición.
=============================================================================
"""

import os
import time
import ctypes
import cv2
import numpy as np
import supervision as sv

# Habilitar soporte High-DPI en Windows (Evita reescalado borroso o pixelado del DWM)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor High-DPI Aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

from src.config import (
    MODELO_PATH, TITULO_VENTANA, CATALOGO_PRECIOS, CLASES_NOMBRES,
    CAMARA_ANCHO_DEFAULT, CAMARA_ALTO_DEFAULT, CAMARA_BUFFER_SIZE,
    TRACK_ACTIVATION_THRESHOLD, LOST_TRACK_BUFFER, MIN_MATCHING_THRESHOLD
)
from src.engine import MotorONNX
from src.state import EstadoSesion
from src.audio import reproducir_sonido_async
from src.receipts import exportar_ticket_digital
from src.ui.layout import ResponsiveLayout


def ejecutar_aplicacion(modo_inicial: str = 'ticket', model_path: str = MODELO_PATH,
                         forzar_cpu: bool = False):
    """
    Inicializa todos los subsistemas y ejecuta el bucle de eventos en tiempo real.
    Abre la ventana en una resolución amplia, nítida y perfectamente proporcionada.
    """
    print("=" * 75)
    print("🚀 INICIANDO TIENDITA INTELIGENTE IA (ALTA DEFINICIÓN HIGH-DPI)")
    print("=" * 75)

    # 1. Inicializar motor de inferencia ONNX (DirectML GPU / CPU)
    motor = MotorONNX(model_path=model_path, forzar_cpu=forzar_cpu)

    # 2. Inicializar rastreador continuo de objetos
    tracker = sv.ByteTrack(
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
        minimum_matching_threshold=MIN_MATCHING_THRESHOLD
    )

    # 3. Anotadores gráficos de alta definición
    corner_annotator = sv.BoxCornerAnnotator(thickness=2, corner_length=18)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.50,
        text_thickness=1,
        text_padding=6,
        border_radius=6
    )

    # 4. Estado de la sesión
    estado = EstadoSesion()
    if modo_inicial in ('ticket', 'inventario', 'completo'):
        estado.modo_idx = ['ticket', 'inventario', 'completo'].index(modo_inicial)

    # 5. Inicializar cámara web
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMARA_ANCHO_DEFAULT)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMARA_ALTO_DEFAULT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMARA_BUFFER_SIZE)

    if not cap.isOpened():
        print("❌ Error crítico: No se pudo abrir la cámara web (índice 0).")
        return

    # Leer un fotograma de prueba para conocer la resolución real de la cámara
    ret, frame_init = cap.read()
    if not ret or frame_init is None:
        cam_w, cam_h = 640, 480
    else:
        cam_h, cam_w = frame_init.shape[:2]

    print(f"📷 Resolución de cámara detectada: {cam_w} x {cam_h}")

    # 6. Calcular resolución y posición óptima de la ventana en pantalla
    total_w, target_h, video_w, sidebar_w, pos_x, pos_y = ResponsiveLayout.calcular_dimensiones(cam_w, cam_h)
    print(f"🖥️ Ventana configurada a: {total_w} x {target_h} (Video: {video_w}px | Sidebar: {sidebar_w}px)")

    # Crear ventana con tamaño explícito y centrada
    cv2.namedWindow(TITULO_VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITULO_VENTANA, total_w, target_h)
    cv2.moveWindow(TITULO_VENTANA, pos_x, pos_y)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 0. Clic para cerrar modal de cobro exitoso si está activo
        if estado.mostrar_modal_cobro:
            estado.mostrar_modal_cobro = False
            reproducir_sonido_async('toggle')
            return

        # 1. Clics dentro del modal de borrado manual
        if estado.mostrar_panel_borrado:
            for hb in estado.hitboxes_modal:
                x1, y1, x2, y2 = hb['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    action = hb['action']
                    prod = hb['prod']
                    if action == 'quitar_1':
                        exito, prec = estado.eliminar_producto(prod)
                        if exito:
                            reproducir_sonido_async('borrar')
                            print(f"🖱️ Clic [-1]: Retirada 1 pza de {prod} (-${prec:.2f}) -> Subtotal: ${estado.subtotal:.2f}")
                    elif action == 'vaciar':
                        exito, monto = estado.vaciar_producto(prod)
                        if exito:
                            reproducir_sonido_async('borrar')
                            print(f"🖱️ Clic [Vaciar]: Retiradas todas las piezas de {prod} (-${monto:.2f}) -> Subtotal: ${estado.subtotal:.2f}")
                    elif action == 'cerrar':
                        estado.mostrar_panel_borrado = False
                        reproducir_sonido_async('toggle')
                    return

        # 2. Clics si la ayuda está abierta
        elif estado.mostrar_ayuda:
            estado.mostrar_ayuda = False
            reproducir_sonido_async('toggle')
            return

        # 3. Clics directos sobre la barra lateral (sidebar)
        else:
            for hb in estado.hitboxes_sidebar:
                x1, y1, x2, y2 = hb['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hb_type = hb['type']
                    prod = hb['prod']
                    if hb_type == 'minus':
                        exito, prec = estado.eliminar_producto(prod)
                        if exito:
                            reproducir_sonido_async('borrar')
                            print(f"🖱️ Clic [-]: Retirada 1 pza de {prod} (-${prec:.2f}) -> Subtotal: ${estado.subtotal:.2f}")
                    elif hb_type == 'plus':
                        exito, prec = estado.agregar_manual(prod)
                        if exito:
                            reproducir_sonido_async('beep')
                            print(f"🖱️ Clic [+]: Agregada 1 pza manual de {prod} (+${prec:.2f}) -> Subtotal: ${estado.subtotal:.2f}")
                    elif hb_type == 'stock_plus':
                        nuevo_stk = estado.ajustar_stock_producto(prod, +1)
                        reproducir_sonido_async('beep')
                        print(f"🖱️ Clic Stock [+1]: {prod} ahora tiene {nuevo_stk} pzas en bodega.")
                    elif hb_type == 'stock_minus':
                        nuevo_stk = estado.ajustar_stock_producto(prod, -1)
                        reproducir_sonido_async('borrar')
                        print(f"🖱️ Clic Stock [-1]: {prod} ahora tiene {nuevo_stk} pzas en bodega.")
                    elif hb_type == 'stock_resurtir_todo':
                        estado.resurtir_stock(10)
                        reproducir_sonido_async('caja')
                        print("🖱️ Clic [Resurtir +10]: +10 piezas agregadas a todo el inventario en stock.json.")
                    elif hb_type == 'btn_modal_borrado':
                        estado.alternar_panel_borrado()
                        reproducir_sonido_async('toggle')
                    return

    cv2.setMouseCallback(TITULO_VENTANA, on_mouse)

    es_pantalla_completa = False

    print("✅ SISTEMA TOTALMENTE OPERATIVO.")
    print("📋 Controles: [Q] Salir | [B] Borrado | [Z] Deshacer | [1-6] Quitar | [C] Cobrar | [L] Resurtir | [M] Modo | [D] -10% | [H] Ayuda")

    fps_hist = 0.0
    t_prev = time.time()
    estabilidad_ids = {}  # Filtro temporal: confirma permanencia antes de cobrar

    while True:
        # -------------------------------------------------------------
        # 1. VERIFICAR SI EL USUARIO PRESIONÓ LA 'X' DE LA VENTANA
        # -------------------------------------------------------------
        try:
            if cv2.getWindowProperty(TITULO_VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                print("🛑 Ventana cerrada mediante la 'X'. Saliendo del programa...")
                break
        except Exception:
            pass

        # -------------------------------------------------------------
        # 2. PROCESAMIENTO DEL FOTOGRAMA EN VIVO
        # -------------------------------------------------------------
        if not estado.pausado:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Aviso: Flujo de video finalizado.")
                break

            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_curr = 1.0 / dt
                fps_hist = (0.90 * fps_hist) + (0.10 * fps_curr) if fps_hist > 0 else fps_curr

            # A. Inferencia de visión artificial sobre la resolución nativa
            detections = motor.inferir(frame)

            # B. Seguimiento multi-objeto continuo
            detections = tracker.update_with_detections(detections)

            # C. Lógica de negocio e inventario
            labels = []
            clases_en_pantalla = set()

            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                ids_visibles = set()
                for xyxy, class_id, tracker_id, conf in zip(
                    detections.xyxy, detections.class_id, detections.tracker_id, detections.confidence
                ):
                    if class_id < len(CLASES_NOMBRES):
                        nombre = CLASES_NOMBRES[class_id]
                        precio = CATALOGO_PRECIOS.get(nombre, 0.0)
                        clases_en_pantalla.add(nombre)
                        ids_visibles.add(tracker_id)

                        # Incrementar contador de cuadros consecutivos que el objeto lleva en pantalla
                        estabilidad_ids[tracker_id] = estabilidad_ids.get(tracker_id, 0) + 1
                        num_cuadros = estabilidad_ids[tracker_id]

                        disponible = estado.obtener_stock_disponible(nombre)

                        if tracker_id in estado.ids_anulados:
                            labels.append(f"#{tracker_id} {nombre} [ANULADO]")
                        elif tracker_id in estado.ids_cobrados:
                            labels.append(f"#{tracker_id} {nombre} [COBRADO]")
                        elif disponible <= 0 and tracker_id not in estado.ids_procesados:
                            labels.append(f"#{tracker_id} {nombre} [SIN STOCK]")
                        elif num_cuadros < 3:
                            labels.append(f"#{tracker_id} {nombre} (fijando...)")
                        else:
                            labels.append(f"#{tracker_id} {nombre} ${precio:.0f} ({conf:.0%})")

                        # Solo cobrar si el objeto permanece estable y no ha sido cobrado ni anulado
                        if num_cuadros >= 3 and tracker_id not in estado.ids_cobrados and tracker_id not in estado.ids_anulados:
                            if disponible <= 0 and tracker_id not in estado.ids_procesados:
                                if num_cuadros == 3:
                                    reproducir_sonido_async('error')
                                    estado.establecer_toast(f"[SIN STOCK] {nombre} agotado en bodega. Venta bloqueada", duracion=3.0)
                                    print(f"⚠️ [BLOQUEADO] #{tracker_id} {nombre} sin stock en bodega.")
                            else:
                                fue_nuevo = estado.registrar_producto(
                                    frame=frame, box=xyxy, nombre=nombre,
                                    precio=precio, conf=conf, tracker_id=tracker_id
                                )
                                if fue_nuevo:
                                    reproducir_sonido_async('beep')
                                    print(f"📦 Escaneado #{tracker_id}: {nombre} (+${precio:.2f}) -> Subtotal: ${estado.subtotal:.2f}")

                # Limpieza de memoria: si un ID cobrado ya no está visible en pantalla, se retira
                estado.ids_cobrados = {i for i in estado.ids_cobrados if i in ids_visibles}

                # Limpieza periódica de memoria de IDs antiguos
                if len(estabilidad_ids) > 100:
                    estabilidad_ids = {k: v for k, v in estabilidad_ids.items() if k in ids_visibles or k in estado.ids_procesados or k in estado.ids_cobrados}
            else:
                ids_visibles = set()
                estado.ids_cobrados.clear()

            # Auto-cerrar modal de cobro si expiró el tiempo
            if estado.mostrar_modal_cobro and time.time() > estado.modal_cobro_expira:
                estado.mostrar_modal_cobro = False

            # D. Renderizado de alta definición y composición visual
            vista_completa = ResponsiveLayout.render(
                frame=frame,
                detections=detections,
                labels=labels,
                corner_annotator=corner_annotator,
                label_annotator=label_annotator,
                estado=estado,
                fps=fps_hist,
                device_name=motor.device_name,
                clases_presentes=clases_en_pantalla,
                video_w=video_w,
                target_h=target_h,
                sidebar_w=sidebar_w
            )

            cv2.imshow(TITULO_VENTANA, vista_completa)

        # -------------------------------------------------------------
        # 3. GESTIÓN INTERACTIVA DE ENTRADAS DE TECLADO
        # -------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        # Manejo de tecla ESC (Cerrar modales o salir del sistema)
        if key == 27:
            if estado.mostrar_modal_cobro:
                estado.mostrar_modal_cobro = False
                reproducir_sonido_async('toggle')
            elif estado.mostrar_panel_borrado:
                estado.mostrar_panel_borrado = False
                reproducir_sonido_async('toggle')
            elif estado.mostrar_ayuda:
                estado.mostrar_ayuda = False
                reproducir_sonido_async('toggle')
            else:
                print("🛑 Finalizando sesión con ESC...")
                break

        # Teclas ENTER (13) o ESPACIO (32) para cerrar modal de cobro si está visible
        elif estado.mostrar_modal_cobro and key in (13, 32):
            estado.mostrar_modal_cobro = False
            reproducir_sonido_async('toggle')

        # Salir limpiamente con Q o X (si no hay modales abiertos)
        elif key in (ord('q'), ord('Q'), ord('x'), ord('X')):
            if estado.mostrar_modal_cobro:
                estado.mostrar_modal_cobro = False
                reproducir_sonido_async('toggle')
            elif estado.mostrar_panel_borrado:
                estado.mostrar_panel_borrado = False
                reproducir_sonido_async('toggle')
            elif estado.mostrar_ayuda:
                estado.mostrar_ayuda = False
                reproducir_sonido_async('toggle')
            else:
                print("🛑 Finalizando sesión...")
                break

        # Alternar Panel de Borrado Manual con B
        elif key in (ord('b'), ord('B')):
            estado.alternar_panel_borrado()
            reproducir_sonido_async('toggle')

        # Deshacer / Borrar último producto escaneado con [Z], [Backspace], [U]
        elif key in (ord('z'), ord('Z'), ord('u'), ord('U'), 8):  # 8 = Backspace en Windows
            exito, nom_b, prec_b = estado.deshacer_ultimo()
            if exito:
                reproducir_sonido_async('borrar')
                print(f"🗑️ Anulado del ticket: {nom_b} (-${prec_b:.2f}) -> Subtotal restante: ${estado.subtotal:.2f}")

        # Borrar producto específico por su número de catálogo [1] al [6]
        elif key in [ord(str(i)) for i in range(1, len(CLASES_NOMBRES) + 1)]:
            idx = key - ord('1')
            prod_sel = CLASES_NOMBRES[idx]
            exito, prec_b = estado.eliminar_producto(prod_sel)
            if exito:
                reproducir_sonido_async('borrar')
                print(f"🗑️ Quitado del ticket [tecla {key - ord('0')}]: {prod_sel} (-${prec_b:.2f}) -> Subtotal restante: ${estado.subtotal:.2f}")

        # Cobro y finalización de compra con C o c
        elif key in (ord('c'), ord('C')):
            if estado.mostrar_modal_cobro:
                # Si el modal ya está en pantalla, pulsar C lo cierra para iniciar la siguiente venta
                estado.mostrar_modal_cobro = False
                reproducir_sonido_async('toggle')
            elif estado.total_items > 0:
                archivo_t = exportar_ticket_digital(
                    inventario=estado.inventario,
                    subtotal=estado.subtotal,
                    descuento_pct=estado.descuento_pct,
                    total_final=estado.total_precio,
                    folio=estado.folio_ticket
                )
                reproducir_sonido_async('caja')
                print(f"💳 Venta completada. Ticket guardado en: {archivo_t}")
                ids_en_escena = ids_visibles if 'ids_visibles' in locals() else None
                estado.finalizar_venta(archivo_ticket=archivo_t, ids_visibles=ids_en_escena)
            else:
                estado.establecer_toast("[AVISO] Carrito vacio. Coloca productos frente a la camara", duracion=2.5)

        # Exportar / Guardar ticket digital con S o s
        elif key in (ord('s'), ord('S')):
            if estado.total_items > 0:
                archivo_t = exportar_ticket_digital(
                    inventario=estado.inventario,
                    subtotal=estado.subtotal,
                    descuento_pct=estado.descuento_pct,
                    total_final=estado.total_precio,
                    folio=estado.folio_ticket
                )
                reproducir_sonido_async('beep')
                estado.establecer_toast("[TICKET] Exportado a carpeta /tickets", duracion=3.0)
                print(f"📄 Recibo emitido: {archivo_t}")
                estado.folio_ticket += 1
            else:
                estado.establecer_toast("[AVISO] No hay productos para generar recibo", duracion=2.5)

        # Alternar descuento del 10% con D o d
        elif key in (ord('d'), ord('D')):
            estado.alternar_descuento()
            reproducir_sonido_async('toggle')

        # Alternar modos de visualización con M o m
        elif key in (ord('m'), ord('M')):
            estado.alternar_modo()
            reproducir_sonido_async('toggle')
            print(f"🔀 Modo: {estado.modo_actual.upper()}")

        # Resurtir / Llenar stock de bodega con L o l
        elif key in (ord('l'), ord('L')):
            estado.resurtir_stock(10)
            reproducir_sonido_async('beep')
            print("📦 [STOCK RESURTIDO] Se agregaron +10 unidades a todos los productos en stock.json.")

        # Reiniciar sesión con R o T
        elif key in (ord('r'), ord('R'), ord('t'), ord('T')):
            estabilidad_ids.clear()
            estado.reiniciar()
            reproducir_sonido_async('reset')
            print("🔄 Sesión e inventario reiniciados.")

        # Pausar / Reanudar transmisión con P o p
        elif key in (ord('p'), ord('P')):
            estado.alternar_pausa()

        # Modal de ayuda con H o Espacio (si modal de cobro no está activo)
        elif key in (ord('h'), ord('H'), ord(' ')):
            if not estado.mostrar_modal_cobro:
                estado.alternar_ayuda()

        # Calibrar umbral de certeza en vivo con [+] y [-]
        elif key in (ord('+'), ord('=')):
            motor.conf_umbral = min(0.95, round(motor.conf_umbral + 0.05, 2))
            estado.establecer_toast(f"[UMBRAL] Certeza calibrada a: {motor.conf_umbral:.0%}", duracion=2.0)
            print(f"🎯 Umbral de Certeza: {motor.conf_umbral:.0%}")

        elif key in (ord('-'), ord('_')):
            motor.conf_umbral = max(0.40, round(motor.conf_umbral - 0.05, 2))
            estado.establecer_toast(f"[UMBRAL] Certeza calibrada a: {motor.conf_umbral:.0%}", duracion=2.0)
            print(f"🎯 Umbral de Certeza: {motor.conf_umbral:.0%}")

        # Alternar Pantalla Completa con F o f
        elif key in (ord('f'), ord('F')):
            es_pantalla_completa = not es_pantalla_completa
            prop = cv2.WINDOW_FULLSCREEN if es_pantalla_completa else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(TITULO_VENTANA, cv2.WND_PROP_FULLSCREEN, prop)
            estado.establecer_toast("[PANTALLA] Modo Pantalla Completa" if es_pantalla_completa else "[VENTANA] Modo Ventana Normal", duracion=2.0)

    # 4. Cierre seguro de recursos
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Aplicación finalizada correctamente.")


if __name__ == '__main__':
    ejecutar_aplicacion(modo_inicial='ticket')
