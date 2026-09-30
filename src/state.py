"""
Módulo de Gestión de Estado de la Sesión.
Centraliza la máquina de estados del carrito, inventario, recortes PiP,
notificaciones Toast y configuraciones interactivas.
"""

import os
import json
import time
import datetime
import cv2
import numpy as np
from .config import CLASES_NOMBRES, MODOS_DISPONIBLES, STOCK_ARCHIVO, STOCK_INICIAL_DEFAULT


class EstadoSesion:
    """Mantiene y administra el estado reactivo de la aplicación."""
    def __init__(self):
        self.stock_tienda = self._cargar_stock()  # Existencias reales de bodega/almacén
        self.inventario = {nombre: 0 for nombre in CLASES_NOMBRES}
        self.ids_procesados = set()
        self.ids_anulados = set()      # IDs cancelados explícitamente para evitar reescaneo involuntario
        self.ids_cobrados = set()      # IDs de la venta recién cobrada que siguen en escena
        self.historial_escaneos = []   # Historial LIFO para posibilitar deshacer/borrar
        self.total_items = 0
        self.subtotal = 0.0
        self.descuento_pct = 0.0  # 0% o 10%
        self.folio_ticket = 1001

        self.modo_idx = 0  # 0: ticket, 1: inventario, 2: completo
        self.pausado = False
        self.mostrar_ayuda = False
        self.mostrar_panel_borrado = False  # Modal interactivo de borrado manual
        self.mostrar_modal_cobro = False    # Modal de confirmación de cobro exitoso
        self.modal_cobro_expira = 0.0       # Tiempo para ocultar automáticamente el modal
        self.datos_ultimo_cobro = {}        # Datos del recibo recién emitido

        # Zonas interactivas clicables (hitboxes de mouse en pantalla)
        self.hitboxes_sidebar = {}
        self.hitboxes_modal = {}

        # Datos para el visor Picture-in-Picture (PiP)
        self.ultimo_crop = None
        self.ultimo_nombre = None
        self.ultimo_precio = 0.0
        self.ultima_conf = 0.0
        self.ultimo_tiempo = ""

        # Notificaciones Toast flotantes
        self.toast_mensaje = "[SISTEMA] Listo. Presiona [H] para ver Ayuda"
        self.toast_expira = time.time() + 4.0

    @property
    def modo_actual(self) -> str:
        """Nombre del modo de visualización en ejecución."""
        return MODOS_DISPONIBLES[self.modo_idx]

    @property
    def total_precio(self) -> float:
        """Total a pagar tras aplicar los descuentos activos."""
        return max(0.0, self.subtotal * (1.0 - (self.descuento_pct / 100.0)))

    @property
    def ultimo_conf(self) -> float:
        """Alias de compatibilidad para ultima_conf."""
        return self.ultima_conf

    def _cargar_stock(self) -> dict:
        """Carga las existencias iniciales desde el archivo stock.json o inicializa con valores por defecto."""
        if os.path.exists(STOCK_ARCHIVO):
            try:
                with open(STOCK_ARCHIVO, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                    for nombre in CLASES_NOMBRES:
                        if nombre not in datos:
                            datos[nombre] = STOCK_INICIAL_DEFAULT.get(nombre, 10)
                    return datos
            except Exception as e:
                print(f"[ADVERTENCIA] Error leyendo {STOCK_ARCHIVO}: {e}. Usando default.")

        stock_nuevo = dict(STOCK_INICIAL_DEFAULT)
        self._guardar_archivo_stock(stock_nuevo)
        return stock_nuevo

    def _guardar_archivo_stock(self, datos: dict):
        """Escribe las existencias a disco en stock.json con formato legible."""
        try:
            with open(STOCK_ARCHIVO, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] No se pudo guardar {STOCK_ARCHIVO}: {e}")

    def guardar_stock(self):
        """Persiste el estado actual de self.stock_tienda a disco."""
        self._guardar_archivo_stock(self.stock_tienda)

    def resurtir_stock(self, cantidad: int = 10):
        """Suma unidades al almacén de la tienda (resurtido rápido) y guarda en disco."""
        for nombre in CLASES_NOMBRES:
            self.stock_tienda[nombre] = self.stock_tienda.get(nombre, 0) + cantidad
        self.guardar_stock()
        self.establecer_toast(f"[RESURTIDO] +{cantidad} unidades agregadas a todo el inventario", duracion=3.5)

    def ajustar_stock_producto(self, nombre: str, delta: int) -> int:
        """Modifica las existencias de almacén de un producto (+1, -1, etc.) y persiste en stock.json."""
        actual = self.stock_tienda.get(nombre, 0)
        nuevo = max(0, actual + delta)
        self.stock_tienda[nombre] = nuevo
        self.guardar_stock()
        signo = f"+{delta}" if delta > 0 else f"{delta}"
        self.establecer_toast(f"[STOCK BODEGA] {nombre}: {signo} (Total: {nuevo} pzas)", duracion=2.0)
        return nuevo

    def obtener_stock_disponible(self, nombre: str) -> int:
        """Retorna la cantidad física disponible en almacén restando lo que ya está en el carrito actual."""
        total_bodega = self.stock_tienda.get(nombre, 0)
        en_carrito = self.inventario.get(nombre, 0)
        return max(0, total_bodega - en_carrito)

    def registrar_producto(self, frame: np.ndarray, box: np.ndarray,
                           nombre: str, precio: float, conf: float, tracker_id: int) -> bool:
        """
        Registra un artículo nuevo en la sesión si su tracker_id no ha sido procesado,
        anulado ni cobrado en la venta previa, y si existen existencias en bodega.
        Retorna True si fue un producto nuevo, o False si ya estaba registrado o sin existencias.
        """
        if tracker_id in self.ids_procesados or tracker_id in self.ids_anulados or tracker_id in self.ids_cobrados:
            return False

        # Validación estricta de existencias en almacén
        disponible = self.obtener_stock_disponible(nombre)
        if disponible <= 0:
            self.establecer_toast(f"[SIN STOCK] {nombre} agotado en bodega. No se puede vender", duracion=3.0)
            return False

        self.ids_procesados.add(tracker_id)
        self.inventario[nombre] = self.inventario.get(nombre, 0) + 1
        self.total_items += 1
        self.subtotal += precio

        # Guardar recorte fotográfico del producto
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_res = None
        tiempo_str = datetime.datetime.now().strftime("%H:%M:%S")
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            crop = frame[y1:y2, x1:x2].copy()
            crop_res = cv2.resize(crop, (110, 105), interpolation=cv2.INTER_LINEAR)
            self.ultimo_crop = crop_res
            self.ultimo_nombre = nombre
            self.ultimo_precio = precio
            self.ultima_conf = conf
            self.ultimo_tiempo = tiempo_str

        # Registrar en el historial de escaneos para permitir reversiones precisas
        self.historial_escaneos.append({
            'nombre': nombre,
            'precio': precio,
            'conf': conf,
            'tracker_id': tracker_id,
            'crop': crop_res,
            'tiempo': tiempo_str
        })

        self.establecer_toast(f"[ESCANEADO] +{nombre} (${precio:.2f})", duracion=2.8)
        return True

    def deshacer_ultimo(self) -> tuple:
        """
        Elimina el último producto escaneado del ticket (operación Undo).
        Retorna (True, nombre, precio) si se eliminó con éxito, o (False, '', 0.0) si no había nada.
        """
        if not self.historial_escaneos or self.total_items <= 0:
            self.establecer_toast("[AVISO] Ticket vacio. No hay articulos para borrar", duracion=2.5)
            return (False, "", 0.0)

        ultimo = self.historial_escaneos.pop()
        nombre = ultimo['nombre']
        precio = ultimo['precio']
        t_id = ultimo.get('tracker_id')

        # Reducir stock y subtotal
        if self.inventario.get(nombre, 0) > 0:
            self.inventario[nombre] -= 1
        self.total_items = max(0, self.total_items - 1)
        self.subtotal = max(0.0, round(self.subtotal - precio, 2))

        # Registrar tracker_id como anulado para evitar que la cámara lo vuelva a detectar de inmediato
        if t_id is not None:
            self.ids_anulados.add(t_id)

        # Restaurar visor PiP al producto inmediatamente anterior en la lista
        if self.historial_escaneos:
            prev = self.historial_escaneos[-1]
            self.ultimo_crop = prev.get('crop')
            self.ultimo_nombre = prev.get('nombre')
            self.ultimo_precio = prev.get('precio', 0.0)
            self.ultima_conf = prev.get('conf', 0.0)
            self.ultimo_tiempo = prev.get('tiempo', '')
        else:
            self.ultimo_crop = None
            self.ultimo_nombre = None
            self.ultimo_precio = 0.0
            self.ultima_conf = 0.0
            self.ultimo_tiempo = ""

        self.establecer_toast(f"[ANULADO] -1 {nombre} (-${precio:.2f})", duracion=3.0)
        return (True, nombre, precio)

    def eliminar_producto(self, nombre: str) -> tuple:
        """
        Resta 1 unidad de un producto específico del ticket e inventario (ej. por tecla numérica).
        Retorna (True, precio) si se eliminó, o (False, 0.0) si no había existencias de ese producto.
        """
        cant = self.inventario.get(nombre, 0)
        if cant <= 0:
            self.establecer_toast(f"[AVISO] No hay '{nombre}' en el ticket para eliminar", duracion=2.5)
            return (False, 0.0)

        from .config import CATALOGO_PRECIOS
        precio = CATALOGO_PRECIOS.get(nombre, 0.0)
        self.inventario[nombre] = max(0, cant - 1)
        self.total_items = max(0, self.total_items - 1)
        self.subtotal = max(0.0, round(self.subtotal - precio, 2))

        # Quitar la última entrada de este producto del historial
        t_id_anulado = None
        for i in range(len(self.historial_escaneos) - 1, -1, -1):
            if self.historial_escaneos[i]['nombre'] == nombre:
                eliminado = self.historial_escaneos.pop(i)
                t_id_anulado = eliminado.get('tracker_id')
                break

        if t_id_anulado is not None:
            self.ids_anulados.add(t_id_anulado)

        # Restaurar visor PiP al último elemento restante
        if self.historial_escaneos:
            prev = self.historial_escaneos[-1]
            self.ultimo_crop = prev.get('crop')
            self.ultimo_nombre = prev.get('nombre')
            self.ultimo_precio = prev.get('precio', 0.0)
            self.ultima_conf = prev.get('conf', 0.0)
            self.ultimo_tiempo = prev.get('tiempo', '')
        else:
            self.ultimo_crop = None
            self.ultimo_nombre = None
            self.ultimo_precio = 0.0
            self.ultima_conf = 0.0
            self.ultimo_tiempo = ""

        self.establecer_toast(f"[QUITADO] -1 {nombre} (-${precio:.2f})", duracion=3.0)
        return (True, precio)

    def agregar_manual(self, nombre: str) -> tuple:
        """Agrega manualmente 1 unidad de un producto al ticket (vía botón + o teclado)."""
        disponible = self.obtener_stock_disponible(nombre)
        if disponible <= 0:
            self.establecer_toast(f"[SIN STOCK] No hay existencias de '{nombre}' para agregar", duracion=2.5)
            return (False, 0.0)

        from .config import CATALOGO_PRECIOS
        precio = CATALOGO_PRECIOS.get(nombre, 0.0)
        self.inventario[nombre] = self.inventario.get(nombre, 0) + 1
        self.total_items += 1
        self.subtotal += precio
        tiempo_str = datetime.datetime.now().strftime("%H:%M:%S")

        self.historial_escaneos.append({
            'nombre': nombre,
            'precio': precio,
            'conf': 1.0,
            'tracker_id': None,
            'crop': None,
            'tiempo': tiempo_str
        })
        self.ultimo_nombre = nombre
        self.ultimo_precio = precio
        self.ultima_conf = 1.0
        self.ultimo_tiempo = tiempo_str
        self.establecer_toast(f"[MANUAL] +{nombre} (${precio:.2f})", duracion=2.5)
        return (True, precio)

    def vaciar_producto(self, nombre: str) -> tuple:
        """Elimina todas las unidades de un producto específico del ticket."""
        cant = self.inventario.get(nombre, 0)
        if cant <= 0:
            self.establecer_toast(f"[AVISO] No hay '{nombre}' en el ticket", duracion=2.0)
            return (False, 0.0)

        from .config import CATALOGO_PRECIOS
        precio = CATALOGO_PRECIOS.get(nombre, 0.0)
        monto = cant * precio
        self.inventario[nombre] = 0
        self.total_items = max(0, self.total_items - cant)
        self.subtotal = max(0.0, round(self.subtotal - monto, 2))

        # Quitar todos los registros de este producto del historial
        for item in self.historial_escaneos:
            if item['nombre'] == nombre and item.get('tracker_id') is not None:
                self.ids_anulados.add(item['tracker_id'])

        self.historial_escaneos = [h for h in self.historial_escaneos if h['nombre'] != nombre]

        if self.historial_escaneos:
            prev = self.historial_escaneos[-1]
            self.ultimo_crop = prev.get('crop')
            self.ultimo_nombre = prev.get('nombre')
            self.ultimo_precio = prev.get('precio', 0.0)
            self.ultima_conf = prev.get('conf', 0.0)
            self.ultimo_tiempo = prev.get('tiempo', '')
        else:
            self.ultimo_crop = None
            self.ultimo_nombre = None
            self.ultimo_precio = 0.0
            self.ultima_conf = 0.0
            self.ultimo_tiempo = ""

        self.establecer_toast(f"[VACIADO] Retiradas todas las piezas de {nombre} (-${monto:.2f})", duracion=3.0)
        return (True, monto)

    def finalizar_venta(self, archivo_ticket: str, ids_visibles: set = None):
        """
        Finaliza la venta actual, guarda los datos del cobro para la interfaz visual,
        marca los productos visibles como ya cobrados para evitar reescaneo inmediato,
        descuenta las piezas vendidas del stock real en bodega y limpia el carrito a cero.
        """
        cant = self.total_items
        tot = self.total_precio
        sub = self.subtotal
        desc = self.descuento_pct
        folio = self.folio_ticket

        # Descontar del stock real de la bodega y persistir
        for nombre, cant_vendida in self.inventario.items():
            if cant_vendida > 0:
                self.stock_tienda[nombre] = max(0, self.stock_tienda.get(nombre, 0) - cant_vendida)
        self.guardar_stock()

        # Marcar los objetos en escena como cobrados para que no se re-escaneen
        if ids_visibles:
            self.ids_cobrados.update(ids_visibles)

        # Datos para el modal visual
        self.datos_ultimo_cobro = {
            'folio': folio,
            'total': tot,
            'subtotal': sub,
            'descuento_pct': desc,
            'articulos': cant,
            'archivo': archivo_ticket,
            'hora': datetime.datetime.now().strftime("%H:%M:%S")
        }
        self.mostrar_modal_cobro = True
        self.modal_cobro_expira = time.time() + 8.0  # Visible 8 segundos o hasta presionar tecla/clic

        # Incrementar folio para el siguiente ticket
        self.folio_ticket += 1

        # Limpiar el carrito de compras e inventario a CERO
        self.inventario = {nombre: 0 for nombre in CLASES_NOMBRES}
        self.total_items = 0
        self.subtotal = 0.0
        self.historial_escaneos.clear()
        self.ultimo_crop = None
        self.ultimo_nombre = None
        self.ultimo_precio = 0.0
        self.ultima_conf = 0.0
        self.ultimo_tiempo = ""
        self.mostrar_panel_borrado = False

        self.establecer_toast(f"[COBRO EXITOSO] Folio #{folio:04d} emitido (${tot:.2f})", duracion=4.0)

    def reiniciar(self):
        """Reinicia el carrito, inventario y registros de seguimiento a cero."""
        self.ids_procesados.clear()
        self.ids_anulados.clear()
        self.ids_cobrados.clear()
        self.historial_escaneos.clear()
        self.inventario = {nombre: 0 for nombre in CLASES_NOMBRES}
        self.total_items = 0
        self.subtotal = 0.0
        self.ultimo_crop = None
        self.ultimo_nombre = None
        self.ultimo_precio = 0.0
        self.ultima_conf = 0.0
        self.ultimo_tiempo = ""
        self.mostrar_panel_borrado = False
        self.mostrar_modal_cobro = False
        self.establecer_toast("[REINICIO] Carrito e inventario en cero", duracion=2.5)

    def alternar_panel_borrado(self):
        """Abre u oculta el modal dedicado de borrado manual."""
        self.mostrar_panel_borrado = not self.mostrar_panel_borrado
        if self.mostrar_panel_borrado:
            self.establecer_toast("[BORRADO MANUAL] Presiona [1-6] o clic para retirar", duracion=3.5)
        else:
            self.establecer_toast("[INFO] Panel de Borrado cerrado", duracion=1.5)

    def alternar_modo(self):
        """Cicla entre los modos: ticket -> inventario -> completo."""
        self.modo_idx = (self.modo_idx + 1) % len(MODOS_DISPONIBLES)
        self.establecer_toast(f"[MODO] Cambiado a: {self.modo_actual.upper()}", duracion=2.0)

    def alternar_descuento(self):
        """Activa o desactiva la promoción del 10% de descuento."""
        self.descuento_pct = 10.0 if self.descuento_pct == 0.0 else 0.0
        if self.descuento_pct > 0:
            self.establecer_toast("[DESC 10%] Descuento de 10% aplicado", duracion=2.5)
        else:
            self.establecer_toast("[DESC 10%] Descuento desactivado", duracion=2.0)

    def alternar_pausa(self):
        """Pausa o reanuda la inferencia y captura."""
        self.pausado = not self.pausado
        self.establecer_toast("[PAUSA] Sistema en pausa" if self.pausado else "[EN VIVO] Sistema reanudado", duracion=2.0)

    def alternar_ayuda(self):
        """Muestra u oculta la ventana modal de ayuda."""
        self.mostrar_ayuda = not self.mostrar_ayuda

    def establecer_toast(self, mensaje: str, duracion: float = 2.5):
        """Despliega un mensaje flotante temporal en pantalla."""
        self.toast_mensaje = mensaje
        self.toast_expira = time.time() + duracion
