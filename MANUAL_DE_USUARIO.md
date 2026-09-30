# 📖 Manual de Usuario y Guía de Operación
## 🏪 Tiendita Inteligente IA — Punto de Venta Autónomo por Visión Artificial

![Estado](https://img.shields.io/badge/Versi%C3%B3n-2.5%20High--DPI-blue?style=for-the-badge)
![Modo](https://img.shields.io/badge/Aceleraci%C3%B3n-GPU%20DirectML-green?style=for-the-badge)
![Auditoría](https://img.shields.io/badge/Seguridad-Local%20%26%20Offline-success?style=for-the-badge)

---

## 📑 Índice de Contenidos
1. [Introducción & Visión General](#1-introducción--visión-general)
2. [Puesta en Marcha y Lanzamiento](#2-puesta-en-marcha-y-lanzamiento)
3. [Anatomía de la Interfaz Visual](#3-anatomía-de-la-interfaz-visual)
4. [Flujo de Trabajo Operativo (Paso a Paso)](#4-flujo-de-trabajo-operativo-paso-a-paso)
5. [Gestión y Corrección de Productos (Borrado Manual)](#5-gestión-y-corrección-de-productos-borrado-manual)
6. [Promociones, Descuentos y Modos de Vista](#6-promociones-descuentos-y-modos-de-vista)
7. [Cobro, Comprobantes y Carpeta de Tickets](#7-cobro-comprobantes-y-carpeta-de-tickets)
8. [Catálogo Oficial de Productos y Códigos](#8-catálogo-oficial-de-productos-y-códigos)
9. [Tabla Maestra de Atajos (Teclado y Ratón)](#9-tabla-maestra-de-atajos-teclado-y-ratón)
10. [Solución de Problemas Frecuentes (FAQ)](#10-solución-de-problemas-frecuentes-faq)
11. [Nota sobre el Uso de Inteligencia Artificial](#11-nota-sobre-el-uso-de-inteligencia-artificial)

---

## 1. Introducción & Visión General

**Tiendita Inteligente IA** es una terminal de punto de venta autónoma (Smart Retail POS) impulsada por visión por computadora en tiempo real. 

El sistema utiliza una cámara web para inspeccionar, clasificar, contar y cobrar productos comerciales mediante una red neuronal profunda **YOLOv8** optimizada con **ONNX Runtime** y aceleración DirectML (GPU). El cajero o cliente simplemente coloca los artículos frente a la cámara y el sistema actualiza automáticamente el ticket sin necesidad de pistolas lectoras de código de barras.

---

## 2. Puesta en Marcha y Lanzamiento

### 2.1 Requisitos de Hardware y Cámara
* **Cámara Web:** Cualquier cámara web USB o integrada (resolución mínima recomendada: 640x480 o 1280x720).
* **Iluminación:** Se recomienda luz blanca o difusa constante para evitar reflejos extremos sobre empaques plásticos o metálicos.
* **Distancia de Escaneo:** Coloca los productos entre **25 cm y 60 cm** del lente de la cámara.

### 2.2 Lanzadores del Sistema
Abre una terminal PowerShell o CMD en la carpeta del proyecto y ejecuta la opción correspondiente:

```bash
# Opción 1: Modo Estándar con Aceleración GPU (Recomendado)
python main.py

# Opción 2: Forzar Modo GPU DirectML (Tarjetas AMD Radeon, Nvidia RTX o Intel Iris)
python FinalGPU.py

# Opción 3: Forzar Modo CPU (Para equipos antiguos sin GPU compatible)
python FinalCPU.py
```

Al iniciar, escucharás un sonido de bienvenida y se abrirá la ventana principal en alta resolución centrada en tu monitor.

---

## 3. Anatomía de la Interfaz Visual

La pantalla está dividida en dos zonas principales calibradas con proporciones áureas:

```text
┌──────────────────────────────────────────────────┬─────────────────────────────┐
│ 🔴 [EN VIVO]  FPS: 22.4  | AMD Radeon RX 6600M   │  TIENDITA INTELIGENTE IA    │
├──────────────────────────────────────────────────┼─────────────────────────────┤
│                                                  │ 📷 ÚLTIMO ARTÍCULO (PiP)    │
│                                                  │ ┌─────────┐ LECHE           │
│                  ÁREA DE VIDEO                   │ │  FOTO   │ $28.00 MXN      │
│                     EN VIVO                      │ │ RECORTE │ Certeza: 98%    │
│                                                  │ └─────────┘ [✓ ESCANEADO]   │
│             (Detección, cajas delimitadoras      ├─────────────────────────────┤
│              y seguimiento multi-objeto)         │ PRODUCTO    P.UNIT AJUSTE TOTAL│
│                                                  │ [1] Aceite   $35.00 [-] x1 [+] $35│
│                                                  │ [2] Atún     $18.00 [-] x0 [+] $0 │
│                                                  │ [3] Leche    $28.00 [-] x2 [+] $56│
│                                                  │ [4] Refresco $22.00 [-] x0 [+] $0 │
│                                                  │ [5] Sopa     $12.00 [-] x1 [+] $12│
│                                                  │ [6] Yogurt   $25.00 [-] x0 [+] $0 │
│                                                  ├─────────────────────────────┤
│                                                  │ Artículos: 4 pzas           │
│                                                  │ Subtotal:   $103.00 MXN     │
│                                                  │ TOTAL:      $103.00 MXN     │
│                                                  ├─────────────────────────────┤
│                                                  │ [B] Borrar • [Z] Deshacer   │
│                                                  │ [C] Cobrar • [S] Ticket     │
└──────────────────────────────────────────────────┴─────────────────────────────┘
```

1. **Visor de Cámara (Izquierda):** Muestra el flujo en vivo. Al detectar un producto, dibuja esquinas tecnológicas de colores con su nombre, precio y porcentaje de certeza.
2. **HUD Superior:** Muestra el estado (`EN VIVO` / `PAUSADO`), los cuadros por segundo (FPS), la tarjeta gráfica en uso y el modo de visualización actual.
3. **Visor PiP (Picture-in-Picture):** Muestra una fotografía instantánea del último producto procesado junto con la hora exacta y el nivel de certeza de la IA.
4. **Tabla de Productos y Botones Táctiles:** Listado de los 6 productos con sus precios unitarios, botones interactivos `[-]` y `[+]`, cantidad acumulada y subtotal por línea.
5. **Panel Financiero:** Muestra el conteo total de artículos, descuentos aplicados y el gran total a cobrar en moneda nacional.
6. **Banda de Controles:** Acceso rápido a las funciones operativas de caja.

---

## 4. Flujo de Trabajo Operativo (Paso a Paso)

### Paso 1: Escaneo de Artículos
1. Sostén el producto frente a la cámara.
2. La IA lo encuadrará de inmediato. Durante los primeros 2 cuadros (~0.10s) la etiqueta dirá `(fijando...)` para garantizar estabilidad y evitar lecturas falsas.
3. Al tercer cuadro consecutivo, el sistema emitirá un **pitido agudo (`beep`)**, registrará el artículo en el carrito, actualizará el visor fotográfico PiP y sumará el importe al subtotal.
4. Retira el producto de la cámara y colócalo en el área de embolsado. Gracias al algoritmo **ByteTrack**, el producto no se cobrará dos veces mientras permanezca en la escena.

### Paso 2: Corrección de Errores (si aplica)
Si un producto se detectó por error o el cliente decide no llevarlo, consulta la sección [5. Gestión y Corrección de Productos](#5-gestión-y-corrección-de-productos-borrado-manual).

### Paso 3: Aplicación de Promociones
Si el cliente cuenta con cupón o tarjeta de cliente frecuente, pulsa la tecla **`[D]`** para aplicar un **10% de descuento directo** sobre el subtotal.

### Paso 4: Finalización y Cobro
Pulsa la tecla **`[C]`**:
1. Sonará el doble timbre comercial de caja registradora.
2. Se generará y guardará el ticket fiscal detallado con folio consecutivo en la carpeta `/tickets`.
3. Aparecerá una notificación verde en pantalla con el número de folio emitido.
4. El carrito se reiniciará automáticamente a cero, dejándolo listo para el siguiente cliente.

---

## 5. Gestión y Corrección de Productos (Borrado Manual)

Uno de los problemas más comunes en caja es cuando un cliente decide no llevar un artículo que ya fue escaneado, o cuando la cámara detectó un objeto no deseado. El sistema ofrece **4 métodos rápidos** para resolver cualquier situación:

### Método A: Deshacer Último Escaneo (Undo Inmediato)
* **Teclas:** **`[Z]`**, **`[Backspace]` (Retroceso)** o **`[U]`**
* **Uso:** Cuando el producto que acabas de escanear fue un error o se registró por accidente.
* **Acción:** Resta inmediatamente 1 unidad del último producto registrado y regresa el visor PiP al producto anterior.

---

### Método B: Retirar un Producto Específico por Teclado `[1 al 6]`
* **Teclas:** **`[1]`** (Aceite), **`[2]`** (Atún), **`[3]`** (Leche), **`[4]`** (Refresco), **`[5]`** (Sopa), **`[6]`** (Yogurt).
* **Uso:** El cliente escaneó 6 productos y dice: *"Ya no me alcanza para la Sopa"*.
* **Acción:** Presionas la tecla **`5`** en tu teclado. El sistema resta exactamente 1 Sopa del ticket y descuenta sus $12.00, sin importar que se haya escaneado hace varios artículos. El resto de la compra permanece intacto.

---

### Método C: Ajuste Táctil Directo con el Ratón `[-]` / `[+]`
* **Uso:** Pantallas táctiles o cajas con mouse.
* **Acción:** 
  * Haz clic en el botón rojo **`[-]`** junto a cualquier producto en la barra lateral para restar 1 pieza.
  * Haz clic en el botón verde **`[+]`** para registrar manualmente una pieza (útil si el empaque está arrugado o no se puede enfocar bien).

---

### Método D: Panel Visual de Borrado y Cancelación `Tecla [B]`
* **Tecla:** **`[B]`** (o haz clic en `[B] Borrado Manual` en el pie de la barra lateral).
* **Acción:** Se abre un modal flotante en el centro de la pantalla que muestra únicamente la lista de productos:
  * **`[-1 QUITAR]`**: Resta una pieza de ese producto.
  * **`[VACIAR TODO]`**: Si el cliente llevaba 5 cajas de leche y cancela todas, este botón retira las 5 de un solo clic.
  * **`[ESC / B]`**: Cierra el panel y regresa al escaneo en vivo.

```text
┌──────────────────────────────────────────────────────────────┐
│        🗑️ PANEL DE CANCELACION Y BORRADO MANUAL               │
├──────────────────────────────────────────────────────────────┤
│ [1] ACEITE ($35.00 c/u)   En ticket: 1 pza   [-1 QUITAR] [V] │
│ [2] ATÚN ($18.00 c/u)     (0 piezas en ticket)               │
│ [3] LECHE ($28.00 c/u)    En ticket: 2 pzas  [-1 QUITAR] [V] │
│ [4] REFRESCO ($22.00 c/u) (0 piezas en ticket)               │
│ [5] SOPA ($12.00 c/u)     En ticket: 3 pzas  [-1 QUITAR] [V] │
│ [6] YOGURT ($25.00 c/u)   (0 piezas en ticket)               │
├──────────────────────────────────────────────────────────────┤
│                [ESC / B] VOLVER AL ESCANEO                   │
└──────────────────────────────────────────────────────────────┘
```

> **🛡️ Protección Anti-Reescaneo:** Al anular o restar un producto, su identificador queda registrado como `[ANULADO]`. Si el producto permanece frente a la cámara, la IA lo marcará en pantalla pero **bloqueará cualquier cobro automático**, evitando cobros duplicados involuntarios.

---

## 6. Promociones, Descuentos y Modos de Vista

* **`[D]` — Descuento del 10%:**
  Aplica instantáneamente un descuento comercial del 10% sobre la base del ticket. El desglose visual reflejará el ahorro en color dorado.
* **`[M]` — Alternar Modos de Interfaz:**
  * **Modo Ticket (Predeterminado):** Diseñado para cobro rápido en punto de venta.
  * **Modo Inventario:** Resalta qué productos del catálogo están presentes frente al mostrador y cuáles faltan en stock.
  * **Modo Completo:** Vista combinada con estadísticas de escaneo.
* **`[P]` — Pausa Temporal:**
  Congela la cámara si el cajero debe ausentarse o limpiar el mostrador sin registrar artículos accidentalmente.
* **`[F]` — Pantalla Completa (Fullscreen):**
  Expande la aplicación a pantalla completa eliminando bordes de ventana para uso en kioscos interactivos.
* **`[+]` / `[-]` — Calibración de Certeza en Vivo:**
  Ajusta el umbral de confianza (+/- 5%) en tiempo real para adaptarse a cambios drásticos de luz (soleado vs noche).

---

## 7. Cobro, Comprobantes y Carpeta de Tickets

Cada venta confirmada con **`[C]`** o respaldada con **`[S]`** genera automáticamente un archivo de texto con desglose fiscal completo en la carpeta local `/tickets`.

### Ejemplo de Recibo Fiscal Generado:

```text
============================================
         TIENDITA INTELIGENTE IA
     SISTEMA POS AUTONOMO POR VISION
       Sucursal Matriz - Terminal #01
============================================
 Folio: #001001     Fecha: 2026-09-29 12:35:10
--------------------------------------------
 CANT  DESCRIPCION      P.UNIT    IMPORTE   
--------------------------------------------
 1     Aceite           $35.00    $35.00    
 2     Leche            $28.00    $56.00    
 1     Sopa             $12.00    $12.00    
--------------------------------------------
 Total de Piezas: 4
--------------------------------------------
 Subtotal Bruto:                  $103.00 MXN
 Descuento (10%):                  -$10.30 MXN
 Subtotal sin IVA:                 $79.91 MXN
 IVA Trasladado (16%):             $12.79 MXN
============================================
 TOTAL A PAGAR:                    $92.70 MXN
============================================
          METODO DE PAGO: EFECTIVO
        PAGADO CON: RECONOCIMIENTO IA

           ||| | ||||| || |||||| | |||
            0123-9874-5561-0042

    ¡GRACIAS POR SU COMPRA EN TIENDITA IA!
     Tecnologia de Vision Artificial YOLO
============================================
```

### 7.1 Modal de Cobro Exitoso y Protección Anti-Reescaneo

Al presionar **`[C]`** (con o sin mayúsculas):
1. **Emisión Inmediata de Comprobante:** Se genera y archiva el ticket fiscal `.txt` en `/tickets`.
2. **Modal Visual Centrado:** Se despliega en pantalla una tarjeta esmeralda con el `TOTAL COBRADO`, desglose de IVA, número de folio y hora exacta.
3. **Vaciado del Carrito:** El subtotal y la tabla lateral se limpian inmediatamente a **$0.00** y **0 artículos**.
4. **Protección Anti-Reescaneo (`[COBRADO]`):** Los productos que siguen reposando en la mesa quedan marcados como `[COBRADO]` en el visor de cámara, impidiendo que el sistema los vuelva a cobrar por error al siguiente cliente hasta que sean retirados físicamente del mostrador.
5. **Cierre:** Puedes cerrar el modal pulsando **`[C]`**, **`[ESPACIO]`**, **`[ESC]`** o haciendo clic en cualquier parte de la pantalla (o esperar 8 segundos a que se cierre solo).

---

### 7.2 Control de Stock Real y Llenado de Inventario (`stock.json`)

El sistema cuenta con un motor de inventario físico permanente desacoplado del carrito de compras:

1. **¿Dónde se guarda el inventario?**
   En el archivo [`stock.json`](file:///c:/Users/mayka/OneDrive/Desktop/Clase%20React%20Native/Tiendita/stock.json) en la raíz del proyecto:
   ```json
   {
       "Aceite": 6,
       "Atun": 10,
       "Leche": 8,
       "Refresco": 12,
       "Sopa": 15,
       "Yogurt": 5
   }
   ```
2. **¿Cómo llenar o modificar el inventario desde la Interfaz?**
   * **Método 1 (Táctil con Ratón en Pantalla):**
     * Presiona **`[M]`** para ingresar al **Modo Inventario**.
     * Haz clic en el botón verde **`[+]`** de cualquier producto para sumarle 1 pieza al almacén, o en **`[-]`** para restarle.
     * En la tarjeta inferior de resumen, haz clic directamente en el botón **`[+10 A TODO (Clic/L)]`** para resurtir 10 piezas a todos los artículos de un solo clic.
   * **Método 2 (En Caliente por Teclado):** Presiona la tecla **`[L]`** en cualquier momento para sumar automáticamente **+10 unidades** a todo el catálogo.
   * **Método 3 (Manual / Configuración):** Abre [`stock.json`](file:///c:/Users/mayka/OneDrive/Desktop/Clase%20React%20Native/Tiendita/stock.json) con el **Bloc de Notas** de Windows, cambia las cantidades al número que tengas físicamente y guarda (`Ctrl + S`).
3. **Protección [SIN STOCK] (Bloqueo de Venta):**
   * Si un producto tiene `0` piezas en almacén (o ya escaneaste todas las piezas disponibles), el sistema **rechaza la venta**.
   * En la cámara aparecerá la etiqueta roja **`#{ID} Producto [SIN STOCK]`**.
   * Sonará una alerta sonora de error y se mostrará un aviso flotante: `[SIN STOCK] Producto agotado en bodega. Venta bloqueada`.
   * El producto **no entrará al ticket ni se sumará al total a pagar**.
4. **Descuento Automático al Cobrar:**
   Al presionar **`[C]`** para cobrar, las piezas vendidas se descuentan automáticamente del almacén y se actualiza `stock.json` al instante.

---

## 8. Catálogo Oficial de Productos y Códigos

| Tecla Rápida | Producto | Categoría | Precio Oficial (MXN) | Tiempo de Estabilidad Requerido |
| :---: | :--- | :--- | :---: | :---: |
| **`[1]`** | **Aceite** | Abarrotes / Cocina | **$35.00** | 3 cuadros (~0.15 seg) |
| **`[2]`** | **Atún** | Enlatados / Pescados | **$18.00** | 3 cuadros (~0.15 seg) |
| **`[3]`** | **Leche** | Lácteos / Bebidas | **$28.00** | 3 cuadros (~0.15 seg) |
| **`[4]`** | **Refresco** | Bebidas / Refrigerados | **$22.00** | 3 cuadros (~0.15 seg) |
| **`[5]`** | **Sopa** | Pastas y Alimentos | **$12.00** | 3 cuadros (~0.15 seg) |
| **`[6]`** | **Yogurt** | Lácteos / Refrigerados | **$25.00** | 3 cuadros (~0.15 seg) |

---

## 9. Tabla Maestra de Atajos (Teclado y Ratón)

| Tecla / Control | Función | Acción Operativa |
| :---: | :--- | :--- |
| **`[B]`** | **Borrado Manual** | Abre o cierra el panel modal interactivo de cancelación. |
| **`[1]` al `[6]`** | **Restar Producto** | Descuenta 1 pieza del producto correspondiente sin alterar los demás. |
| **Clic `[-]`** | **Restar con Ratón** | Resta 1 unidad del producto seleccionado en la tabla lateral. |
| **Clic `[+]`** | **Sumar con Ratón** | Agrega manualmente 1 unidad del producto seleccionado. |
| **`[Z]` / `[Backspace]`** | **Deshacer Último** | Anula inmediatamente el último artículo ingresado al ticket. |
| **`[C]`** | **Cobrar Venta** | Emite ticket fiscal, emite sonido de caja, descuenta stock de bodega y resetea. |
| **`[L]`** | **Resurtir Stock** | Agrega +10 piezas a todos los productos en bodega (`stock.json`). |
| **`[S]`** | **Guardar Ticket** | Guarda copia de seguridad del ticket actual en `/tickets`. |
| **`[D]`** | **Descuento 10%** | Activa o desactiva la promoción del 10% sobre el total. |
| **`[M]`** | **Modo de Vista** | Cambia entre vista Ticket, Inventario y Completo. |
| **`[R]` / `[T]`** | **Reiniciar Todo** | Vacía el carrito por completo y reinicia IDs a cero. |
| **`[P]`** | **Pausar Cámara** | Congela / reanuda temporalmente el flujo de video y detección. |
| **`[F]`** | **Pantalla Completa** | Alterna entre ventana estándar y modo pantalla completa (Kiosco). |
| **`[+]` / `[-]`** | **Calibrar Certeza** | Aumenta o disminuye en vivo el umbral de admisión de la IA (+/- 5%). |
| **`[H]` / `[ESPACIO]`** | **Ventana de Ayuda** | Muestra u oculta la guía de arquitectura y atajos en pantalla. |
| **`[ESC]`** | **Cerrar / Salir** | Cierra cualquier modal abierto; si no hay modales, cierra el sistema. |
| **`[Q]` / `[X]`** | **Salir del Sistema** | Finaliza la aplicación de forma segura liberando la GPU y cámara. |

---

## 10. Solución de Problemas Frecuentes (FAQ)

### ¿Qué hago si la cámara confunde un objeto del fondo con un producto?
1. **Solución Inmediata:** Pulsa la tecla **`[+]`** dos veces para elevar el umbral de certeza (por ejemplo, de 80% a 90%). La IA se volverá más estricta y solo admitirá coincidencias casi perfectas.
2. **Eliminar el error:** Pulsa **`[Z]`** para deshacer el producto si fue el último, o presiona su número **`[1-6]`** si ya pasaron varios artículos.

### ¿Por qué el producto no se escanea de inmediato al presentarlo?
El sistema cuenta con un **filtro de estabilidad temporal de 3 cuadros (~0.15 segundos)**. Esto es intencional para evitar que movimientos bruscos o productos que solo van pasando por delante se cobren accidentalmente. Sostén el producto quieto medio segundo frente al lente.

### ¿Se pueden cobrar dos productos iguales a la vez?
Sí. Puedes colocar dos leches en pantalla al mismo tiempo. El rastreador **ByteTrack** les asignará dos IDs distintos (ejemplo `#12 Leche` y `#13 Leche`) y registrará ambas piezas en el ticket con sendos pitidos de confirmación.

### ¿Dónde consulto las ventas anteriores?
Todos los comprobantes emitidos quedan respaldados cronológicamente en la carpeta:
```text
Tiendita/tickets/ticket_YYYYMMDD_HHMMSS_fXXXX.txt
```
Puedes abrirlos con cualquier editor de texto o enviarlos a una impresora térmica POS.

---

## 11. Nota sobre el Uso de Inteligencia Artificial

> [!NOTE]
> **Transparencia y Metodología:**
> Tanto la redacción y organización pedagógica de este manual de usuario como la optimización visual y ergonómica de las interfaces gráficas (UI/UX) contaron con el soporte de herramientas de **Inteligencia Artificial (IA)**, empleadas para:
> * **Estructuración y Redacción de la Documentación:** Diseñar explicaciones paso a paso, tablas de referencia rápida y protocolos de resolución de incidencias.
> * **Diseño de Interfaces de Usuario (UI/UX):** Perfeccionar la estética obsidian de la terminal, la jerarquía visual de los elementos en pantalla, el contraste cromático y la disposición ergonómica de los botones y modales de interacción.

---

> 📄 **Términos de Propiedad Intelectual:** Consulta el archivo [LICENSE.md](LICENSE.md) para conocer la titularidad de los derechos de autor y las licencias de las bibliotecas de código abierto integradas.

