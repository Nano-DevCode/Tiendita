"""
Definición de Tema y Estilos Visuales de Alta Definición (High-DPI Theme).
Paleta cromática moderna estilo Fintech / Cyber POS, gestión de fuentes TrueType
(Segoe UI) con renderizado sub-píxel anti-aliasing para eliminar toda pixelación.
"""

import os
import cv2
from PIL import ImageFont


class FontManager:
    """Caché de fuentes TrueType de alta definición para evitar lecturas de disco."""
    _cache = {}

    @classmethod
    def get_font(cls, size: int, bold: bool = False):
        key = (size, bold)
        if key not in cls._cache:
            if bold:
                rutas = [
                    'C:/Windows/Fonts/segoeuib.ttf',
                    'C:/Windows/Fonts/arialbd.ttf',
                    'C:/Windows/Fonts/calibrib.ttf'
                ]
            else:
                rutas = [
                    'C:/Windows/Fonts/segoeui.ttf',
                    'C:/Windows/Fonts/arial.ttf',
                    'C:/Windows/Fonts/calibri.ttf'
                ]

            font_path = None
            for r in rutas:
                if os.path.exists(r):
                    font_path = r
                    break

            try:
                if font_path:
                    cls._cache[key] = ImageFont.truetype(font_path, size)
                else:
                    cls._cache[key] = ImageFont.load_default()
            except Exception:
                cls._cache[key] = ImageFont.load_default()

        return cls._cache[key]


class Theme:
    """Paleta cromática moderna en formatos RGB (PIL) y BGR (OpenCV)."""

    # --- COLORES RGB PARA PILLOW (TrueType Anti-Aliased Rendering) ---
    RGB_BG_DARK       = (11, 15, 23)        # Slate profundo
    RGB_CARD_BG       = (19, 27, 42)        # Tarjetas elevadas
    RGB_CARD_BORDER   = (35, 48, 71)        # Bordes sutiles
    RGB_CARD_ACTIVE   = (28, 40, 62)        # Resaltado de fila activa
    RGB_TOTAL_BG_TOP  = (6, 78, 59)         # Verde esmeralda superior
    RGB_TOTAL_BG_BOT  = (4, 120, 87)        # Verde esmeralda inferior

    RGB_CYAN          = (0, 229, 255)       # Cian eléctrico neón
    RGB_GREEN         = (16, 185, 129)      # Menta esmeralda
    RGB_MINT_LIGHT    = (52, 211, 153)      # Verde menta claro
    RGB_GOLD          = (245, 158, 11)      # Ámbar dorado
    RGB_CORAL         = (239, 68, 68)       # Coral / Rojo suave
    RGB_PURPLE        = (168, 85, 247)      # Púrpura neón

    RGB_TEXT_TITLE    = (255, 255, 255)     # Blanco puro
    RGB_TEXT_BODY     = (241, 245, 249)     # Blanco grisáceo suave
    RGB_TEXT_MUTED    = (148, 163, 184)     # Gris intermedio
    RGB_TEXT_DIM      = (100, 116, 139)     # Gris apagado

    # --- COLORES BGR PARA OPENCV (HUD en video) ---
    BGR_BG_DARK       = (23, 15, 11)
    BGR_CARD_BG       = (42, 27, 19)
    BGR_CARD_BORDER   = (71, 48, 35)
    BGR_CYAN          = (255, 229, 0)
    BGR_GREEN         = (129, 185, 16)
    BGR_GOLD          = (11, 158, 245)
    BGR_CORAL         = (68, 68, 239)
    BGR_TEXT_TITLE    = (255, 255, 255)
    BGR_TEXT_MUTED    = (184, 163, 148)

    FONT = cv2.FONT_HERSHEY_DUPLEX
    LINE_AA = cv2.LINE_AA
