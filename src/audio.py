"""
Servicio de Notificaciones de Audio Asíncrono.
Provee retroalimentación sonora típica de un punto de venta (escaneo y caja)
utilizando la biblioteca estándar de Windows sin bloquear el bucle de video.
"""

import threading

try:
    import winsound
    HAY_SONIDO = True
except ImportError:
    HAY_SONIDO = False


def reproducir_sonido_async(tipo: str = 'beep'):
    """
    Despacha la reproducción de un sonido en un hilo daemon secundario.
    Tipos soportados:
    - 'beep': Pitido agudo corto al detectar un producto nuevo.
    - 'caja': Doble campanilla de caja registradora al completar venta.
    - 'reset': Tono grave suave para indicar reinicio de datos.
    - 'toggle': Tono medio para cambios de modo o activación de descuento.
    """
    if not HAY_SONIDO:
        return

    def _worker():
        try:
            if tipo == 'beep':
                winsound.Beep(2100, 80)
            elif tipo == 'caja':
                winsound.Beep(1300, 90)
                winsound.Beep(1850, 130)
            elif tipo in ('delete', 'borrar'):
                winsound.Beep(950, 70)
                winsound.Beep(650, 100)
            elif tipo == 'reset':
                winsound.Beep(850, 120)
            elif tipo == 'toggle':
                winsound.Beep(1600, 60)
            elif tipo in ('error', 'alerta'):
                winsound.Beep(450, 150)
                winsound.Beep(350, 200)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()
