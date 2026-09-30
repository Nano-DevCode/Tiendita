"""
Módulo de Configuración y Constantes Globales.
Define parámetros de inferencia, catálogo oficial, rutas y opciones de ventana.
"""

import os

# --- RUTAS DE ARCHIVOS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(BASE_DIR, 'MiModelo_YOLO_BEST.onnx')
CARPETA_TICKETS = os.path.join(BASE_DIR, 'tickets')

# --- PARÁMETROS DE VISIÓN E INFERENCIA (ESTÁNDAR COMERCIAL RIGUROSO) ---
TAMANO_MODELO = 640
CONF_UMBRAL = 0.80          # Estándar riguroso de identificación (>=80% de certeza)
IOU_UMBRAL = 0.45

# Filtros geométricos de validación física de productos (Retail Standards)
MIN_AREA_PCT = 0.015        # Área mínima: al menos 1.5% del fotograma (descarta teclas, monedas)
MAX_AREA_PCT = 0.85         # Área máxima: descarta fondos que cubran toda la pantalla
MIN_ASPECT_RATIO = 0.35     # Relación de aspecto mínima (alto/ancho)
MAX_ASPECT_RATIO = 2.40     # Relación máxima (descarta objetos alargados como teclados o tiras)

# --- PARÁMETROS DE SEGUIMIENTO (BYTE TRACK) ---
TRACK_ACTIVATION_THRESHOLD = 0.70  # Requiere certeza sólida para iniciar seguimiento
LOST_TRACK_BUFFER = 30
MIN_MATCHING_THRESHOLD = 0.75
CUADROS_CONFIRMACION = 4           # Requiere 4 cuadros consecutivos estables para cobro

# --- CATÁLOGO OFICIAL DE PRODUCTOS Y PRECIOS UNITARIOS (MXN) ---
CATALOGO_PRECIOS = {
    'Aceite':   35.00,
    'Atun':     18.50,
    'Leche':    28.00,
    'Refresco': 15.00,
    'Sopa':     12.00,
    'Yogurt':   10.00
}

CLASES_NOMBRES = ['Aceite', 'Atun', 'Leche', 'Refresco', 'Sopa', 'Yogurt']

# Paleta de colores distintivos BGR por categoría de producto
COLORES_CLASES = {
    'Aceite':   (0, 165, 255),    # Naranja cálido
    'Atun':     (238, 187, 0),    # Cian tecnológico
    'Leche':    (255, 230, 128),  # Azul cielo suave
    'Refresco': (68, 68, 255),    # Coral vibrante
    'Sopa':     (0, 215, 255),    # Amarillo oro
    'Yogurt':   (210, 105, 235),  # Violeta neón
}

# --- CONFIGURACIÓN DE VENTANA Y CÁMARA ---
TITULO_VENTANA = "Tiendita Inteligente IA - POS & Inventario (Portfolio Edition)"
CAMARA_ANCHO_DEFAULT = 1280
CAMARA_ALTO_DEFAULT = 720
CAMARA_BUFFER_SIZE = 1

# Modos disponibles del sistema
MODOS_DISPONIBLES = ['ticket', 'inventario', 'completo']

# --- CONFIGURACIÓN DE STOCK / EXISTENCIAS DE ALMACÉN ---
STOCK_ARCHIVO = os.path.join(BASE_DIR, 'stock.json')
STOCK_INICIAL_DEFAULT = {
    'Aceite':   6,
    'Atun':     10,
    'Leche':    8,
    'Refresco': 12,
    'Sopa':     15,
    'Yogurt':   5
}

