"""
Módulo de Generación y Exportación de Tickets Digitales.
Crea comprobantes de venta detallados con desglose fiscal, descuentos y formato POS.
"""

import os
import datetime
from .config import CARPETA_TICKETS, CATALOGO_PRECIOS, CLASES_NOMBRES


def exportar_ticket_digital(inventario: dict, subtotal: float, descuento_pct: float,
                            total_final: float, folio: int) -> str:
    """
    Genera un comprobante de venta en formato texto (.txt) listo para impresoras
    térmicas o almacenamiento digital.

    Retorna la ruta absoluta del archivo generado.
    """
    os.makedirs(CARPETA_TICKETS, exist_ok=True)
    ahora = datetime.datetime.now()
    fecha_str = ahora.strftime("%Y-%m-%d %H:%M:%S")
    nombre_archivo = os.path.join(CARPETA_TICKETS, f"ticket_{ahora.strftime('%Y%m%d_%H%M%S')}_f{folio}.txt")

    iva_pct = 0.16
    descuento_monto = subtotal * (descuento_pct / 100.0)
    base_imponible = subtotal - descuento_monto
    iva_monto = base_imponible * (iva_pct / (1.0 + iva_pct))
    subtotal_sin_iva = base_imponible - iva_monto

    lineas = [
        "=" * 44,
        "         TIENDITA INTELIGENTE IA",
        "     SISTEMA POS AUTONOMO POR VISION",
        "       Sucursal Matriz - Terminal #01",
        "=" * 44,
        f" Folio: #{folio:06d}     Fecha: {fecha_str}",
        "-" * 44,
        f" {'CANT':<5} {'DESCRIPCION':<16} {'P.UNIT':<9} {'IMPORTE':<10}",
        "-" * 44
    ]

    total_piezas = 0
    for prod in CLASES_NOMBRES:
        cant = inventario.get(prod, 0)
        if cant > 0:
            precio_u = CATALOGO_PRECIOS.get(prod, 0.0)
            imp = cant * precio_u
            total_piezas += cant
            lineas.append(f" {cant:<5} {prod:<16} ${precio_u:<8.2f} ${imp:<9.2f}")

    lineas.extend([
        "-" * 44,
        f" Total de Piezas: {total_piezas}",
        "-" * 44,
        f" Subtotal Bruto:                 ${subtotal:>9.2f} MXN",
    ])

    if descuento_pct > 0:
        lineas.append(f" Descuento ({descuento_pct:.0f}%):               -${descuento_monto:>9.2f} MXN")

    lineas.extend([
        f" Subtotal sin IVA:               ${subtotal_sin_iva:>9.2f} MXN",
        f" IVA Trasladado (16%):           ${iva_monto:>9.2f} MXN",
        "=" * 44,
        f" TOTAL A PAGAR:                  ${total_final:>9.2f} MXN",
        "=" * 44,
        "          METODO DE PAGO: EFECTIVO",
        "        PAGADO CON: RECONOCIMIENTO IA",
        "",
        "           ||| | ||||| || |||||| | |||",
        "            0123-9874-5561-0042",
        "",
        "    ¡GRACIAS POR SU COMPRA EN TIENDITA IA!",
        "     Tecnologia de Vision Artificial YOLO",
        "=" * 44
    ])

    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("\n".join(lineas) + "\n")

    return nombre_archivo
